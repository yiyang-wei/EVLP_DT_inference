from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from .reformat import *


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

        self.hourly_display_df = None
        self.image_pc_display_df = None
        self.protein_display_df = None
        self.transcriptomics_display_df = None

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

        self.predictions_display = None

    def load_input_data(self, dfs):
        self.hourly_display_df = dfs[hourly_lung_function_sheet]
        self.image_pc_display_df = dfs[lung_image_sheet]
        self.protein_display_df = dfs[protein_sheet]
        self.transcriptomics_display_df = dfs[transcriptomics_sheet]

        hourly_calculated_delta = hourly_calculate_delta(self.hourly_display_df)
        hourly_with_calculated_display_df = pd.concat([self.hourly_display_df, hourly_calculated_delta], axis=0)
        self.hourly_input_data = pd.DataFrame([hourly_display_to_input(hourly_with_calculated_display_df)])

        pc_model_input_h1_df, pc_model_input_h3_df = image_pc_display_to_input(self.image_pc_display_df)
        self.pc_1h_input_data = pd.DataFrame([pc_model_input_h1_df])
        self.pc_3h_input_data = pd.DataFrame([pc_model_input_h3_df])

        protein_slope_df = calculate_protein_slopes(self.protein_display_df)
        protein_model_input_df = pd.DataFrame([protein_display_to_input(self.protein_display_df)])
        protein_slope_input_df = pd.DataFrame([protein_slope_display_to_input(protein_slope_df)])
        self.protein_input_data = pd.concat([protein_model_input_df, protein_slope_input_df], axis=1)
        self.transcriptomics_input_data = pd.DataFrame([transcriptomics_display_to_input(self.transcriptomics_display_df)])

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

    def run(self):
        self.hourly_dynamic_inference()
        self.hourly_static_inference()
        self.image_pc_dynamic_inference()
        self.image_pc_static_inference()
        self.protein_dynamic_inference()
        self.protein_static_inference()
        self.transcriptomics_dynamic_inference()
        self.transcriptomics_static_inference()

    def get_pred_display(self):
        hourly_pred_h2_display = hourly_input_to_display(self.hourly_pred_h2.iloc[0])
        hourly_pred_h3_static_display = hourly_input_to_display(self.hourly_pred_h3_static.iloc[0])
        hourly_pred_h3_dynamic_display = hourly_input_to_display(self.hourly_pred_h3_dynamic.iloc[0])
        hourly_predictions_display = self.hourly_display_df[["2nd Hour", "3rd Hour"]].add_prefix("Observed ")
        hourly_predictions_display["Predicted 2nd Hour"] = hourly_pred_h2_display["2nd Hour"].astype(float).round(1)
        hourly_predictions_display["Static Predicted 3rd Hour"] = hourly_pred_h3_static_display["3rd Hour"].astype(float).round(1)
        hourly_predictions_display["Dynamic Predicted 3rd Hour"] = hourly_pred_h3_dynamic_display["3rd Hour"].astype(float).round(1)

        image_pc_pred_static_display = image_pc_input_to_display(None, self.pc_pred_static.iloc[0])
        image_pc_pred_dynamic_display = image_pc_input_to_display(None, self.pc_pred_dynamic.iloc[0])
        image_pc_predictions_display = self.image_pc_display_df[["3rd Hour"]].add_prefix("Observed ")
        image_pc_predictions_display["Static Predicted 3rd Hour"] = image_pc_pred_static_display["3rd Hour"].astype(float)
        image_pc_predictions_display["Dynamic Predicted 3rd Hour"] = image_pc_pred_dynamic_display["3rd Hour"].astype(float)

        protein_pred_h2_display = protein_input_to_display(self.protein_pred_h2.iloc[0])
        protein_pred_h3_static_display = protein_input_to_display(self.protein_pred_h3_static.iloc[0])
        protein_pred_h3_dynamic_display = protein_input_to_display(self.protein_pred_h3_dynamic.iloc[0])
        protein_predictions_display = self.protein_display_df[["2nd Hour", "3rd Hour"]].add_prefix("Observed ")
        protein_predictions_display["Predicted 2nd Hour"] = protein_pred_h2_display["2nd Hour"].astype(float).round(1)
        protein_predictions_display["Static Predicted 3rd Hour"] = protein_pred_h3_static_display["3rd Hour"].astype(float).round(1)
        protein_predictions_display["Dynamic Predicted 3rd Hour"] = protein_pred_h3_dynamic_display["3rd Hour"].astype(float).round(1)

        transcriptomics_pred_static_display = transcriptomics_input_to_display(self.transcriptomics_pred_static.iloc[0])
        transcriptomics_pred_dynamic_display = transcriptomics_input_to_display(self.transcriptomics_pred_dynamic.iloc[0])
        transcriptomics_predictions_display = self.transcriptomics_display_df[["Target"]].add_prefix("Observed ")
        transcriptomics_predictions_display["Static Predicted Target"] = transcriptomics_pred_static_display["Target"].astype(int)
        transcriptomics_predictions_display["Dynamic Predicted Target"] = transcriptomics_pred_dynamic_display["Target"].astype(int)

        self.predictions_display = {
            "Hourly Lung Function Prediction": hourly_predictions_display,
            "Lung X-ray Image Prediction": image_pc_predictions_display,
            "Protein Prediction": protein_predictions_display,
            "Transcriptomics Prediction": transcriptomics_predictions_display
        }
