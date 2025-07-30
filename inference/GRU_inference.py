import torch
import numpy as np
import pandas as pd
from GRU.util.baselines import average_of_tail
from GRU.forecasting_pipeline import param_idx
from GRU.GRU import GRU
import pathlib

static_setups = ["A1F50_A2F50", "A1F50PA2F50_A3F50", "A1F50_A3F50"]
dynamic_setups = ["A1F50A2F50_A3F50"]
parameter_names = {
    "Dy_comp": "Dynamic Compliance (mL/cmH₂O)",
    "P_peak": "Peak Airway Pressure (cmH₂O)",
    "P_mean": "Mean Airway Pressure (cmH₂O)",
    "Ex_vol": "Expiratory Volume (mL)"
}   # Order must match the order of model input

device = torch.device("cpu")

def read_simulated_data(file_path):
    xl = pd.ExcelFile(file_path)

    simulated_data = {}
    sheet_map = {
        "A1F50": "1Hr Per-breath Data",
        "A2F50": "2Hr Per-breath Data",
        "A3F50": "3Hr Per-breath Data"
    }
    for key, sheet in sheet_map.items():
        if sheet in xl.sheet_names:
            simulated_data[key] = np.array(pd.read_excel(file_path, sheet_name=sheet)[list(parameter_names.values())]).reshape(1, 50, 4)
        else:
            simulated_data[key] = None
    return simulated_data

class TimeSeriesInference:

    def __init__(self, model_folder):
        self.model_folder = pathlib.Path(model_folder)

        self.a1 = None
        self.a2 = None
        self.a3 = None

        self.pred_a2 = None
        self.static_pred_a3 = None
        self.dynamic_pred_a3 = None

    def load_model(self, stage: str, param: str) -> torch.nn.Module:
        path = self.model_folder / "GRU" / stage / param / "seed_42_multivariate" / "locked_model.pt"
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model

    def load_input_data(self, dfs):
        if "1Hr Per-breath Data" in dfs:
            self.a1 = dfs["1Hr Per-breath Data"][list(parameter_names.values())]
        if "2Hr Per-breath Data" in dfs:
            self.a2 : pd.DataFrame= dfs["2Hr Per-breath Data"][list(parameter_names.values())]
            self.pred_a2 = self.a2.copy()
            self.pred_a2[:] = np.nan
        if "3Hr Per-breath Data" in dfs:
            self.a3 = dfs["3Hr Per-breath Data"][list(parameter_names.values())]
            self.static_pred_a3 = self.a3.copy()
            self.static_pred_a3[:] = np.nan
            self.dynamic_pred_a3 = self.a3.copy()
            self.dynamic_pred_a3[:] = np.nan

    def inference_all_parameters(self, input_x, model_setup):
        results = {}
        for param, param_name in parameter_names.items():
            model = self.load_model(model_setup, param)
            avg_of_x = average_of_tail(input_x)[:, :, param_idx[param]: param_idx[param] + 1]
            with torch.no_grad():
                x = torch.tensor(input_x, dtype=torch.float32).to(device, non_blocking=True)
                y_preds = model(x, None).detach().cpu().numpy()

            y_preds = y_preds + avg_of_x
            results[param_name] = y_preds.flatten()
        return results

    def static_inference(self):
        if self.a1 is None:
            raise ValueError("First hour time series data is missing, cannot perform static inference.")

        pred_a2_results = self.inference_all_parameters(self.a1.to_numpy().reshape(1, 50, 4), "A1F50_A2F50")
        for param, pred in pred_a2_results.items():
            self.pred_a2[param] = pred

        a1_pa2 = np.concatenate([self.a1.to_numpy().reshape(1, 50, 4), self.pred_a2.to_numpy().reshape(1, 50, 4)], axis=1)
        static_pred_a3_results = self.inference_all_parameters(a1_pa2, "A1F50_A3F50")

        for param, pred in static_pred_a3_results.items():
            self.static_pred_a3[param] = pred

    def dynamic_inference(self):
        if self.a1 is None or self.a2 is None:
            raise ValueError("First or second hour time series data is missing, cannot perform dynamic inference.")

        a1_a2 = np.concatenate([self.a1.to_numpy().reshape(1, 50, 4), self.a2.to_numpy().reshape(1, 50, 4)], axis=1)
        dynamic_pred_a3_results = self.inference_all_parameters(a1_a2, "A1F50A2F50_A3F50")
        for param, pred in dynamic_pred_a3_results.items():
            self.dynamic_pred_a3[param] = pred


def timeseries_inference(
        model_folder,
        output_folder,
        file_path,
        mode="static"  # or "dynamic", "static+dynamic"
):
    model_folder = pathlib.Path(model_folder)
    output_folder = pathlib.Path(output_folder)

    # Read in the synthetic data
    simulated_data = read_simulated_data(file_path)
    case_id = int(file_path.stem.replace("DT Lung Demo Case ", ""))

    if mode == "static":
        if simulated_data["A1F50"] is None:
            raise ValueError("A1F50 data is missing, cannot perform static inference.")
        setups = static_setups
    elif mode == "dynamic":
        if simulated_data["A1F50"] is None or simulated_data["A2F50"] is None:
            raise ValueError("A1F50 or A2F50 data is missing, cannot perform dynamic inference.")
        setups = dynamic_setups
    elif mode == "static+dynamic":
        if simulated_data["A1F50"] is None:
            raise ValueError("A1F50 data is missing, cannot perform static or dynamic inference.")
        setups = static_setups
        if simulated_data["A2F50"] is None:
            print("A2F50 data is missing, cannot perform dynamic inference.")
        else:
            setups += dynamic_setups
    
    # Do inference for the setups and each parameter
    device = torch.device("cpu")

    for setup in setups:
        for param in parameter_names:
            datasets = {
                "A1F50_A2F50": (simulated_data["A1F50"], simulated_data["A2F50"][:, :, param_idx[param]: param_idx[param] +1] if simulated_data["A2F50"] is not None else None),
                "A1F50_A3F50": (simulated_data["A1F50"], simulated_data["A3F50"][:, :, param_idx[param]: param_idx[param] +1] if simulated_data["A3F50"] is not None else None),
            }
            if mode == "dynamic" or mode == "static+dynamic":
                datasets["A1F50A2F50_A3F50"] = (np.concatenate([simulated_data["A1F50"], simulated_data["A2F50"]], axis=1), simulated_data["A3F50"][:, :, param_idx[param]: param_idx[param] +1] if simulated_data["A3F50"] is not None else None)

            # Load the saved model
            model_setup_to_use = setup
            if setup == "A1F50PA2F50_A3F50":
                model_setup_to_use = "A1F50A2F50_A3F50"
                datasets["A1F50PA2F50_A3F50"] = (np.concatenate([simulated_data["A1F50"], simulated_data["PA2F50"]], axis=1), simulated_data["A3F50"][:, :, param_idx[param]: param_idx[param] +1] if simulated_data["A3F50"] is not None else None)

            Model_Folder = model_folder / "GRU" / model_setup_to_use / param / "seed_42_multivariate"
            model = torch.load(Model_Folder / f"locked_model.pt", map_location=device, weights_only=False)
            model.eval()

            avg_of_X = average_of_tail(datasets[setup][0])[:, :, param_idx[param]: param_idx[param] +1]
            true_Y = datasets[setup][1]

            with torch.no_grad():
                x = torch.tensor(datasets[setup][0], dtype=torch.float32).to(device, non_blocking=True)
                y_preds = model(x, None).detach().cpu().numpy()

            y_preds = y_preds + avg_of_X

            results = pd.DataFrame()
            results["case_id"] = np.repeat([case_id], 50)
            if true_Y is not None:
                results["true_y"] = true_Y.flatten()
            results["pred_y"] = y_preds.flatten()

            result_save_path = output_folder / "High-resolution time series" / setup
            result_save_path.mkdir(parents=True, exist_ok=True)

            results.to_csv(result_save_path / f"{param}_true_pred.csv", index=False)

            # Save A2 predictions to be used for A1F50PA2F50_A3F50 setup
            if setup == "A1F50_A2F50":
                np.save(result_save_path / f"{param}_y_pred.npy", y_preds)
                if param == list(parameter_names.keys())[-1]:
                    # Add "PA2F50" to arrays
                    PA2F50_y_pred = []
                    for p in parameter_names:
                        y_pred = np.load(result_save_path / f"{p}_y_pred.npy")
                        PA2F50_y_pred.append(y_pred)
                    PA2F50_y_pred = np.concatenate(PA2F50_y_pred, axis=2)
                    simulated_data["PA2F50"] = PA2F50_y_pred.reshape(1, 50, 4)
