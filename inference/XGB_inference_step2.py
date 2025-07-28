from XGB.inference import *
import pathlib


def example_hourly_inferences_step_2(model_folder, data_folder, output_folder):
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
    new_pc_1h_data_path = data_folder / "PC1h_data_simulated.csv"
    new_transcriptomics_x_data_path = data_folder / "transcriptomics1_data_simulated.csv"
    new_transcriptomics_y_data_path = data_folder / "transcriptomics2_data_simulated.csv"

    # Inference settings
    inference_suffix = " inference"
    models = ["XGBoost"]

    hourly_true_h1 = pd.read_csv(os.path.join(hourly_save_folder, f"H1_to_H2{inference_suffix}/test_X.csv"), index_col=0)
    hourly_pred_h2 = pd.read_csv(os.path.join(hourly_save_folder, f"H1_to_H2{inference_suffix}/predicted_Y_XGBoost.csv"), index_col=0)
    hourly_pred_h3 = pd.read_csv(os.path.join(hourly_save_folder, f"H1_to_H3{inference_suffix}/predicted_Y_XGBoost.csv"), index_col=0)

    new_pc_1h_data = pd.read_csv(new_pc_1h_data_path, index_col=1)
    new_pc_1h_data = new_pc_1h_data.drop(new_pc_1h_data.columns[0], axis=1)


    new_pc_3h_prediction = pd.DataFrame()
    for file in os.listdir(pca_3h_prediction_save_folder):
        if not file.endswith(".csv"):
            continue
        feature_name = file.split(".")[0]
        new_pc_3h_prediction[feature_name] = pd.read_csv(os.path.join(pca_3h_prediction_save_folder, file), index_col=0)["y_pred"]


    new_pc_1h_data.columns = [col + "_x" for col in new_pc_1h_data.columns]
    new_transcriptomics_x_data = pd.read_csv(new_transcriptomics_x_data_path, index_col=1)
    new_transcriptomics_x_data = new_transcriptomics_x_data.drop(new_transcriptomics_x_data.columns[0], axis=1)
    new_transcriptomics_y_data = pd.read_csv(new_transcriptomics_y_data_path, index_col=1)
    new_transcriptomics_y_data = new_transcriptomics_y_data.drop(new_transcriptomics_y_data.columns[0], axis=1)
    transcriptomics_inference_new_cases_static(
        new_data=pd.concat([new_transcriptomics_x_data, new_transcriptomics_y_data, hourly_true_h1, hourly_pred_h2, hourly_pred_h3, new_pc_1h_data, new_pc_3h_prediction], axis=1),
        model_folder=transcriptomics_model_folder,
        save_folder=transcriptomics_save_folder,
        suffix=inference_suffix,
        models=models,
    )




if __name__ == "__main__":
    Model_folder = pathlib.Path("../Model")
    Data_folder = pathlib.Path("../Data")
    Output_folder = pathlib.Path("../Output")
    example_hourly_inferences_step_2(Model_folder, Data_folder, Output_folder)