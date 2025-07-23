import pathlib
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

def save_results(save_folder, predicted_Y, true_Y):
    save_folder.mkdir(parents=True, exist_ok=True)
    predicted_Y.to_csv(save_folder / "predicted_Y.csv", index=True)
    true_Y.to_csv(save_folder / "true_Y.csv", index=True)

def hourly_inferences_step_1(model_folder, data_folder, output_folder):

    model_folder = pathlib.Path(model_folder)
    data_folder = pathlib.Path(data_folder)
    output_folder = pathlib.Path(output_folder)

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
    pred_steen_h2 = pd.read_csv(hourly_save_folder / "STEEN_H1_to_H2_H3" / f"predicted_Y.csv", index_col=0)
    true_hourly_h3 = pd.read_csv(hourly_save_folder / "H1_H2_to_H3" / "true_Y.csv", index_col=0)
    true_steen_h3 = pd.read_csv(hourly_save_folder / "STEEN_H1_H2_to_H3" / "true_Y.csv", index_col=0)
    hourly_h1_pred_h2 = pd.concat([hourly_h1, pred_hourly_h2, pred_steen_h2], axis=1)
    pred_h1_h2_to_h3 = predict_with_model(hourly_model_folder / "H1_H2_to_H3", hourly_h1_pred_h2)
    save_results(hourly_save_folder / "H1_pred_H2_to_H3", pred_h1_h2_to_h3, true_hourly_h3)
    pred_steen_h1_h2_to_h3 = predict_with_model(hourly_model_folder / "STEEN_H1_H2_to_H3", hourly_h1_pred_h2)
    save_results(hourly_save_folder / "STEEN_H1_pred_H2_to_H3", pred_steen_h1_h2_to_h3, true_steen_h3)


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
    model_folder = pathlib.Path(model_folder)
    data_folder = pathlib.Path(data_folder)
    output_folder = pathlib.Path(output_folder)

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
    Model_folder = pathlib.Path("../Model")
    Data_folder = pathlib.Path("../Data")
    Output_folder = pathlib.Path("../Output")
    hourly_inferences_step_1(Model_folder, Data_folder, Output_folder)
    # hourly_inferences_step_2(Model_folder, Data_folder, Output_folder)
