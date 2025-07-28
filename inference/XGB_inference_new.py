from pathlib import Path
import numpy as np
import pandas as pd
import pickle


def load_model(path):
    with open(path, "rb") as file:
        model = pickle.load(file)
    model.enable_categorical = True
    booster = model.get_booster()
    feature_names = list(booster.feature_names)
    return model, feature_names

def predict_with_model(folder, data):
    param_preds = {}
    for model_file in folder.glob("*.pkl"):
        param = model_file.stem
        model, x_columns = load_model(model_file)
        x_data = data[x_columns]
        prediction = model.predict(x_data)
        param_preds[param] = prediction
    predicted_Y = pd.DataFrame(param_preds, index=data.index)
    return predicted_Y

def save_results(save_folder, predicted_Y, true_Y=None):
    save_folder.mkdir(parents=True, exist_ok=True)
    predicted_Y.to_csv(save_folder / "predicted_Y.csv", index=True)
    if true_Y is not None:
        true_Y.to_csv(save_folder / "true_Y.csv", index=True)


class XGBInference:

    def __init__(self, model_folder):
        self.model_folder = Path(model_folder)
        self.hourly_model_folder = self.model_folder / "XGB" / "Hourly"
        self.protein_model_folder = self.model_folder / "XGB" / "Protein"
        self.transcriptomics_model_folder = self.model_folder / "XGB" / "Transcriptomics"
        self.image_pc_model_folder = self.model_folder / "XGB_PC"

        self.hourly_input_data = None
        self.protein_input_data = None
        self.pc_1h_input_data = None
        self.pc_3h_input_data = None
        self.transcriptomics_input_data = None

        self.hourly_pred_h2 = None
        self.hourly_h1_pred_h2 = None
        self.hourly_pred_h3_static = None
        self.hourly_pred_h3_dynamic = None

        self.protein_pred_h2 = None
        self.protein_pred_h3_static = None
        self.protein_pred_h3_dynamic = None

        self.transcriptomics_pred_static = None
        self.transcriptomics_pred_dynamic = None

        self.pc_pred_static = None
        self.pc_pred_dynamic = None


    def hourly_dynamic_inference(self):
        hourly_h1_h2_to_h3_model_folder = self.hourly_model_folder / "H1_H2_to_H3"
        self.hourly_pred_h3_dynamic = predict_with_model(hourly_h1_h2_to_h3_model_folder, self.hourly_input_data)
        return self.hourly_pred_h3_dynamic

    def hourly_static_inference(self):
        hourly_h1_to_h2_model_folder = self.hourly_model_folder / "H1_to_H2"
        hourly_h1_h2_to_h3_model_folder = self.hourly_model_folder / "H1_H2_to_H3"
        hourly_h1 = self.hourly_input_data.loc[:, self.hourly_input_data.columns.str.startswith("70_")]
        self.hourly_pred_h2 = predict_with_model(hourly_h1_to_h2_model_folder, self.hourly_input_data)
        self.hourly_h1_pred_h2 = pd.concat([hourly_h1, self.hourly_pred_h2], axis=1)
        self.hourly_pred_h3_static = predict_with_model(hourly_h1_h2_to_h3_model_folder, self.hourly_h1_pred_h2)
        return self.hourly_pred_h2, self.hourly_pred_h3_static

    def protein_dynamic_inference(self):
        protein_h1_h2_to_h3_model_folder = self.protein_model_folder / "H1_H2_to_H3"
        protein_dynamic_input = pd.concat([self.protein_input_data, self.hourly_input_data, self.pc_1h_input_data], axis=1)
        self.protein_pred_h3_dynamic = predict_with_model(protein_h1_h2_to_h3_model_folder, protein_dynamic_input)
        return self.protein_pred_h3_dynamic

    def protein_static_inference(self):
        protein_h1_to_h2_model_folder = self.protein_model_folder / "H1_to_H2"
        protein_h1_pred_h2_to_h3_model_folder = self.protein_model_folder / "H1_pred_H2_to_H3"
        protein_static_input = pd.concat([self.protein_input_data, self.hourly_h1_pred_h2, self.pc_1h_input_data], axis=1)
        self.protein_pred_h2 = predict_with_model(protein_h1_to_h2_model_folder, protein_static_input)
        self.protein_pred_h3_static = predict_with_model(protein_h1_pred_h2_to_h3_model_folder, protein_static_input)
        return self.protein_pred_h2, self.protein_pred_h3_static

    def transcriptomics_dynamic_inference(self):
        transcriptomics_dynamic_model_folder = self.transcriptomics_model_folder / "dynamic_forecasting"
        pc_1h_input_data = self.pc_1h_input_data.add_suffix("_x")
        pc_3h_input_data = self.pc_3h_input_data.add_suffix("_y")
        transcriptomics_dynamic_input = pd.concat([self.transcriptomics_input_data, self.hourly_input_data, pc_1h_input_data, pc_3h_input_data], axis=1)
        self.transcriptomics_pred_dynamic = predict_with_model(transcriptomics_dynamic_model_folder, transcriptomics_dynamic_input)
        return self.transcriptomics_pred_dynamic

    def transcriptomics_static_inference(self):
        transcriptomics_static_model_folder = self.transcriptomics_model_folder / "static_forecasting"
        pc_1h_input_data = self.pc_1h_input_data.add_suffix("_x")
        pc_3h_pred_static = self.pc_pred_static.add_suffix("_predictions")
        transcriptomics_static_input = pd.concat([self.transcriptomics_input_data, self.hourly_h1_pred_h2, self.hourly_pred_h3_static, pc_1h_input_data, pc_3h_pred_static], axis=1)
        self.transcriptomics_pred_static = predict_with_model(transcriptomics_static_model_folder, transcriptomics_static_input)
        return self.transcriptomics_pred_static

    def image_pc_dynamic_inference(self):
        feat_dir = self.image_pc_model_folder / 'results_true_hyper'
        model_dir = self.image_pc_model_folder / 'models_true'

        tab_data_df = self.hourly_input_data.copy()
        image_pc1h = self.pc_1h_input_data.add_prefix("1h_")
        image_pc3h = self.pc_3h_input_data.add_prefix("3h_")
        merged_df = pd.concat([tab_data_df, image_pc1h, image_pc3h], axis=1)

        y_columns = [f'3h_feature_pca_{i}' for i in range(10)]
        preds = pd.DataFrame(columns=y_columns)
        for col_name in y_columns:
            pca_num = int(col_name.replace('3h_feature_pca_', ''))
            features_path = feat_dir / f'top_features_pca_{pca_num}.txt'
            with open(features_path, 'r') as f:
                features = f.read().splitlines()

            X_test = merged_df[features].values

            with open(model_dir / f'{col_name}_model.pkl', 'rb') as file:
                pipeline = pickle.load(file)

            y_pred = pipeline.predict(X_test)
            preds[col_name] = y_pred

        self.pc_pred_dynamic = preds
        self.pc_pred_dynamic.index = self.pc_1h_input_data.index
        return self.pc_pred_dynamic

    def image_pc_static_inference(self):
        feat_dir = self.image_pc_model_folder / 'results_true_hyper'
        model_dir = self.image_pc_model_folder / 'models_pred'

        image_pc1h = self.pc_1h_input_data.add_prefix("1h_")
        image_pc3h = self.pc_3h_input_data.add_prefix("3h_")
        merged_df = pd.concat([self.hourly_h1_pred_h2, image_pc1h, image_pc3h], axis=1)

        y_columns = [f'3h_feature_pca_{i}' for i in range(10)]
        preds = pd.DataFrame(columns=y_columns)
        for col_name in y_columns:
            pca_num = int(col_name.replace('3h_feature_pca_', ''))
            features_path = feat_dir / f'top_features_pca_{pca_num}.txt'
            with open(features_path, 'r') as f:
                features = f.read().splitlines()

            X_test = merged_df[features].values

            with open(model_dir / f'{col_name}_model.pkl', 'rb') as file:
                pipeline = pickle.load(file)

            y_pred = pipeline.predict(X_test)
            preds[col_name] = y_pred
        self.pc_pred_static = preds
        self.pc_pred_static.index = self.pc_1h_input_data.index
        return self.pc_pred_static

def hourly_inferences_step_1(model_folder, data_folder, output_folder):

    model_folder = Path(model_folder)
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)

    # Saved model folders (input folders)
    hourly_model_folder = model_folder / "XGB" / "Hourly"
    protein_model_folder = model_folder / "XGB" / "Protein"
    transcriptomics_model_folder = model_folder / "XGB" / "Transcriptomics"

    # New inference save folders (output folders)
    hourly_save_folder = output_folder / "Hourly"
    protein_save_folder = output_folder / "Protein"
    transcriptomics_save_folder = output_folder / "Transcriptomics"

    # Inference input data paths
    new_hourly_data_path = data_folder / "hourly_data_simulated.csv"
    new_edema_data_path = data_folder / "edema_data_simulated.csv"
    new_pc_1h_data_path = data_folder / "PC1h_data_simulated.csv"
    new_pc_3h_data_path = data_folder / "PC3h_data_simulated.csv"
    new_protein_data_path = data_folder / "protein_data_simulated_withslopes.csv"
    new_transcriptomics_x_data_path = data_folder / "transcriptomics1_data_simulated.csv"
    new_transcriptomics_y_data_path = data_folder / "transcriptomics2_data_simulated.csv"

    # Hourly inference
    new_hourly_data = pd.read_csv(new_hourly_data_path, index_col=1)
    new_hourly_data = new_hourly_data.drop(new_hourly_data.columns[0], axis=1)
    new_edema_data = pd.read_csv(new_edema_data_path, index_col=1)
    new_edema_data = new_edema_data.drop(new_edema_data.columns[0], axis=1)
    new_hourly_data = pd.concat([new_hourly_data, new_edema_data], axis=1)

    for hourly_setup in hourly_model_folder.iterdir():
        if not hourly_setup.is_dir():
            continue
        predicted_Y = predict_with_model(hourly_setup, new_hourly_data)
        true_Y = new_hourly_data[predicted_Y.columns]
        save_results(hourly_save_folder / hourly_setup.name, predicted_Y, true_Y)

    # h1 + pred_h2 to h3 inference
    hourly_h1 = new_hourly_data.loc[:, new_hourly_data.columns.str.startswith("70_")]
    pred_hourly_h2 = pd.read_csv(hourly_save_folder / "H1_to_H2" / f"predicted_Y.csv", index_col=0)
    true_hourly_h3 = pd.read_csv(hourly_save_folder / "H1_H2_to_H3" / "true_Y.csv", index_col=0)
    hourly_h1_pred_h2 = pd.concat([hourly_h1, pred_hourly_h2], axis=1)
    pred_h1_h2_to_h3 = predict_with_model(hourly_model_folder / "H1_H2_to_H3", hourly_h1_pred_h2)
    save_results(hourly_save_folder / "H1_pred_H2_to_H3", pred_h1_h2_to_h3, true_hourly_h3)


    # Protein dynamic inference
    new_pc_1h_data = pd.read_csv(new_pc_1h_data_path, index_col=1)
    new_pc_1h_data = new_pc_1h_data.drop(new_pc_1h_data.columns[0], axis=1)
    new_protein_data = pd.read_csv(new_protein_data_path, index_col=1)
    new_protein_data = new_protein_data.drop(new_protein_data.columns[0], axis=1)
    protein_dynamic_input = pd.concat([new_protein_data, new_hourly_data, new_pc_1h_data], axis=1)
    protein_static_input = pd.concat([new_protein_data, hourly_h1_pred_h2, new_pc_1h_data], axis=1)

    for protein_setup in protein_model_folder.iterdir():
        if protein_setup.name == "H1_pred_H2_to_H3":
            predicted_Y = predict_with_model(protein_setup, protein_static_input)
        else:
            predicted_Y = predict_with_model(protein_setup, protein_dynamic_input)
        true_Y = new_protein_data[predicted_Y.columns]
        save_results(protein_save_folder / protein_setup.name, predicted_Y, true_Y)

    # Transcriptomics dynamic inference
    new_pc_3h_data = pd.read_csv(new_pc_3h_data_path, index_col=1)
    new_pc_3h_data = new_pc_3h_data.drop(new_pc_3h_data.columns[0], axis=1)
    new_pc_1h_data.columns = [col + "_x" for col in new_pc_1h_data.columns]
    new_pc_3h_data.columns = [col + "_y" for col in new_pc_3h_data.columns]
    new_transcriptomics_x_data = pd.read_csv(new_transcriptomics_x_data_path, index_col=1)
    new_transcriptomics_x_data = new_transcriptomics_x_data.drop(new_transcriptomics_x_data.columns[0], axis=1)
    new_transcriptomics_y_data = pd.read_csv(new_transcriptomics_y_data_path, index_col=1)
    new_transcriptomics_y_data = new_transcriptomics_y_data.drop(new_transcriptomics_y_data.columns[0], axis=1)
    new_transcriptomics_data = pd.concat([new_transcriptomics_x_data, new_transcriptomics_y_data], axis=1)
    transcriptomics_dynamic_input = pd.concat([new_transcriptomics_data, new_hourly_data, new_pc_1h_data, new_pc_3h_data], axis=1)
    transcriptomics_setup = transcriptomics_model_folder / "dynamic_forecasting"
    predicted_Y = predict_with_model(transcriptomics_setup, transcriptomics_dynamic_input)
    true_Y = new_transcriptomics_y_data[predicted_Y.columns]
    save_results(transcriptomics_save_folder / transcriptomics_setup.name, predicted_Y, true_Y)

def hourly_inferences_step_2(model_folder, data_folder, output_folder):
    model_folder = Path(model_folder)
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)

    # Saved model folders (input folders)
    transcriptomics_model_folder = model_folder / "XGB" / "Transcriptomics"

    # Saved inference folders (input folders)
    hourly_save_folder = output_folder / "Hourly"
    pca_3h_prediction_save_folder = output_folder / "ImagePC" / "Dynamic"

    # New inference save folders (output folders)
    transcriptomics_save_folder = output_folder / "Transcriptomics"

    # Inference input data paths
    new_hourly_data_path = data_folder / "hourly_data_simulated.csv"
    new_pc_1h_data_path = data_folder / "PC1h_data_simulated.csv"
    new_transcriptomics_x_data_path = data_folder / "transcriptomics1_data_simulated.csv"
    new_transcriptomics_y_data_path = data_folder / "transcriptomics2_data_simulated.csv"

    hourly_pred_h2 = pd.read_csv(hourly_save_folder / f"H1_to_H2" / "predicted_Y.csv", index_col=0)
    hourly_pred_h3 = pd.read_csv(hourly_save_folder / f"H1_pred_H2_to_H3" / "predicted_Y.csv", index_col=0)

    new_hourly_data = pd.read_csv(new_hourly_data_path, index_col=1)
    new_hourly_data = new_hourly_data.drop(new_hourly_data.columns[0], axis=1)
    hourly_h1 = new_hourly_data.loc[:, new_hourly_data.columns.str.startswith("70_")]
    new_pc_1h_data = pd.read_csv(new_pc_1h_data_path, index_col=1)
    new_pc_1h_data = new_pc_1h_data.drop(new_pc_1h_data.columns[0], axis=1)

    new_pc_3h_prediction = pd.DataFrame()
    for file in pca_3h_prediction_save_folder.iterdir():
        if not file.suffix == ".csv":
            continue
        feature_name = file.stem
        new_pc_3h_prediction[feature_name] = pd.read_csv(pca_3h_prediction_save_folder / file.name, index_col=0)["y_pred"]

    new_pc_1h_data.columns = [col + "_x" for col in new_pc_1h_data.columns]
    new_transcriptomics_x_data = pd.read_csv(new_transcriptomics_x_data_path, index_col=1)
    new_transcriptomics_x_data = new_transcriptomics_x_data.drop(new_transcriptomics_x_data.columns[0], axis=1)
    new_transcriptomics_y_data = pd.read_csv(new_transcriptomics_y_data_path, index_col=1)
    new_transcriptomics_y_data = new_transcriptomics_y_data.drop(new_transcriptomics_y_data.columns[0], axis=1)
    transcriptomics_static_input = pd.concat([new_transcriptomics_x_data, new_transcriptomics_y_data, hourly_h1, hourly_pred_h2, hourly_pred_h3, new_pc_1h_data, new_pc_3h_prediction], axis=1)
    transcriptomics_setup = transcriptomics_model_folder / "static_forecasting"
    predicted_Y = predict_with_model(transcriptomics_setup, transcriptomics_static_input)
    true_Y = new_transcriptomics_y_data[predicted_Y.columns]
    save_results(transcriptomics_save_folder / transcriptomics_setup.name, predicted_Y, true_Y)


if __name__ == "__main__":
    Model_folder = Path("../Model")
    Data_folder = Path("../Data")
    Output_folder = Path("../Output")
    hourly_inferences_step_1(Model_folder, Data_folder, Output_folder)
    # hourly_inferences_step_2(Model_folder, Data_folder, Output_folder)
