from inference.reformat import *
from inference.XGB_inference_new import XGBInference
from inference.visualization import *

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from huggingface_hub import snapshot_download


@st.cache_data
def load_excel(file_path):
    return pd.read_excel(file_path, sheet_name=None, index_col=0)

@st.cache_data
def load_excel_binary(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def download_huggingface(data_folder, model_folder):
    model_repo_id = "SageLabUHN/DT_Lung"
    data_repo_id = "SageLabUHN/DT_Lung_Demo_Data"
    with st.spinner(f"Downloading dataset from huggingface.co/datasets/{data_repo_id}"):
        snapshot_download(data_repo_id, repo_type="dataset", local_dir=data_folder, local_dir_use_symlinks=False)
    st.success(f"Successfully downloaded dataset from huggingface.co/datasets/{data_repo_id}")
    with st.spinner(f"Downloading model from huggingface.co/{model_repo_id}"):
        snapshot_download(model_repo_id, local_dir=model_folder, local_dir_use_symlinks=False)
    st.success(f"Successfully downloaded model from huggingface.co/{model_repo_id}")

@st.cache_resource
def run_xgb_inference(model_folder, demo_dfs):
    xgb_inference = XGBInference(model_folder)
    xgb_inference.load_input_data(demo_dfs)
    xgb_inference.run()
    xgb_inference.get_pred_display()
    return xgb_inference


def main():

    st.set_page_config(
        page_title="EVLP DT",
        page_icon=":material/respiratory_rate:",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    data_folder = pathlib.Path("Data")
    model_folder = pathlib.Path("Model")
    output_folder = pathlib.Path("Output")
    output_folder.mkdir(parents=True, exist_ok=True)

    demo_case_prefix = "DT Lung Demo Case "

    st.title(":material/respiratory_rate: Digital Twin of Ex-Vivo Human Lungs ")

    st.subheader("Step 0: Download Models and Data")

    if not data_folder.exists() or not model_folder.exists():
        download_huggingface(data_folder, model_folder)
    else:
        st.success("Models and data already downloaded. You can redownload them if needed.")

    redownload_huggingface = st.button(
        label="Redownload Models and Data",
        icon=":material/refresh:",
        use_container_width=True,
    )
    if redownload_huggingface:
        download_huggingface(data_folder, model_folder)

    st.subheader("Step 1: Prepare Data")
    data_source = st.radio(
        "Use your own data or our demo data?",
        options=["Use Your Own Data", "Use Demo Data"],
        index=0
    )

    demo_files = data_folder.glob(demo_case_prefix + "*")
    demo_names = [file.stem for file in demo_files if file.is_file()]
    selected_demo_case = st.selectbox(
        label="Select a Demo Case",
        options=demo_names,
        index=0,
    )
    selected_demo_id = int(selected_demo_case.replace(demo_case_prefix, ""))

    demo_dfs = load_excel(data_folder / f"{selected_demo_case}.xlsx")
    hourly_display_df = demo_dfs[hourly_lung_function_sheet]
    image_pc_display_df = demo_dfs[lung_image_sheet]
    protein_display_df = demo_dfs[protein_sheet]
    transcriptomics_display_df = demo_dfs[transcriptomics_sheet]
    time_series_display_dfs = [
        demo_dfs[per_breath_h1_sheet],
        demo_dfs[per_breath_h2_sheet],
        demo_dfs[per_breath_h3_sheet]
    ]

    with (st.expander(f"Data for {selected_demo_case}", expanded=True)):
        (
            hourly_display_tab,
            image_pc_display_tab,
            protein_display_tab,
            transcriptomics_display_tab,
            time_series_a1_display_tab,
            time_series_a2_display_tab,
            time_series_a3_display_tab,
        ) = st.tabs([
            "Hourly Lung Function Data",
            "Lung Image Data",
            "Protein Data",
            "Transcriptomics Data",
            "1st Hour per-breath Data",
            "2nd Hour per-breath Data",
            "3rd Hour per-breath Data"
        ])
        hourly_display_tab.dataframe(hourly_display_df)
        image_pc_display_tab.dataframe(image_pc_display_df)
        protein_display_tab.dataframe(protein_display_df)
        transcriptomics_display_tab.dataframe(transcriptomics_display_df)
        time_series_a1_display_tab.dataframe(time_series_display_dfs[0])
        time_series_a2_display_tab.dataframe(time_series_display_dfs[1])
        time_series_a3_display_tab.dataframe(time_series_display_dfs[2])

    st.subheader("Step 2: Run Inference")

    prediction_save_path = output_folder / f"{selected_demo_case} predictions.xlsx"

    predictions_display = None
    saved_predictions = None
    if prediction_save_path.exists():
        st.info(f"Predictions for {selected_demo_case} already exist. You can view the results below or re-run the inference.")
        predictions_display = load_excel(prediction_save_path)

    run_inference = st.button(
        label="Run Inference",
        icon=":material/play_arrow:",
        use_container_width=True
    )

    xgb_inference = None
    if run_inference:
        xgb_inference = run_xgb_inference(model_folder, demo_dfs)
        st.success("Inference completed successfully!")
        with pd.ExcelWriter(prediction_save_path) as writer:
            for sheet_name, df in xgb_inference.predictions_display.items():
                df.to_excel(writer, sheet_name=sheet_name)
        predictions_display = xgb_inference.predictions_display

    st.subheader("Step 3: View Results")
    if predictions_display is not None:
        (
            hourly_pred_tab,
            image_pc_pred_tab,
            protein_pred_tab,
            transcriptomics_pred_tab,
        ) = st.tabs([
            "Hourly Lung Function Predictions",
            "Lung X-ray Image Predictions",
            "Protein Predictions",
            "Transcriptomics Predictions"
        ])

        hourly_pred_tab.dataframe(predictions_display["Hourly Lung Function Prediction"])
        hourly_pred_tab.plotly_chart(
            hourly_all_features_line_plot(predictions_display["Hourly Lung Function Prediction"]),
            use_container_width=True,
            # theme=None

        )
        image_pc_pred_tab.dataframe(predictions_display["Lung X-ray Image Prediction"])
        protein_pred_tab.dataframe(predictions_display["Protein Prediction"])
        transcriptomics_pred_tab.dataframe(predictions_display["Transcriptomics Prediction"])

    if prediction_save_path.exists():
        saved_predictions = load_excel_binary(prediction_save_path)

    st.subheader("Step 4: Download Predictions")
    st.download_button(
        label="Download Predictions",
        data=saved_predictions,
        file_name=f"{selected_demo_case} predictions.xlsx",
        mime="application/vnd.ms-excel",
        disabled=saved_predictions is None,
        use_container_width=True
    )



if __name__ == "__main__":
    main()
