from inference.reformat import *
from inference.XGB_inference_new import XGBInference
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from huggingface_hub import snapshot_download


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name=None, index_col=0)

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
    hourly_calculated_delta = hourly_calculate_delta(demo_dfs[hourly_lung_function_sheet])
    hourly_with_calculated_display_df = pd.concat([demo_dfs[hourly_lung_function_sheet], hourly_calculated_delta], axis=0)
    hourly_model_input_df = pd.DataFrame([hourly_display_to_input(hourly_with_calculated_display_df)])
    pc_model_input_h1_df, pc_model_input_h3_df = image_pc_display_to_input(demo_dfs[lung_image_sheet])
    pc_model_input_h1_df = pd.DataFrame([pc_model_input_h1_df])
    pc_model_input_h3_df = pd.DataFrame([pc_model_input_h3_df])
    protein_slope_df = calculate_protein_slopes(demo_dfs[protein_sheet])
    protein_model_input_df = pd.DataFrame([protein_display_to_input(demo_dfs[protein_sheet])])
    protein_slope_input_df = pd.DataFrame([protein_slope_display_to_input(protein_slope_df)])
    protein_with_slope_model_input_df = pd.concat([protein_model_input_df, protein_slope_input_df], axis=1)
    transcriptomics_model_input_df = pd.DataFrame([transcriptomics_display_to_input(demo_dfs[transcriptomics_sheet])])
    xgb_inference = XGBInference(model_folder)
    xgb_inference.hourly_input_data = hourly_model_input_df
    xgb_inference.pc_1h_input_data = pc_model_input_h1_df
    xgb_inference.pc_3h_input_data = pc_model_input_h3_df
    xgb_inference.protein_input_data = protein_with_slope_model_input_df
    xgb_inference.transcriptomics_input_data = transcriptomics_model_input_df
    xgb_inference.hourly_dynamic_inference()
    xgb_inference.hourly_static_inference()
    xgb_inference.image_pc_dynamic_inference()
    xgb_inference.image_pc_static_inference()
    xgb_inference.protein_dynamic_inference()
    xgb_inference.protein_static_inference()
    xgb_inference.transcriptomics_dynamic_inference()
    xgb_inference.transcriptomics_dynamic_inference()
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

    demo_case_prefix = "DT Lung Demo Case "

    st.title(":material/respiratory_rate: Ex-Vivo Lung Perfusion Digital Twin")

    st.subheader("Step 0: Download Models and Data")

    if not data_folder.exists() or not model_folder.exists():
        download_huggingface(data_folder, model_folder)

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

    demo_dfs = load_data(data_folder / f"{selected_demo_case}.xlsx")
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
    run_inference = st.button(
        label="Run Inference",
        icon=":material/play_arrow:",
        use_container_width=True
    )

    xgb_inference = None
    if run_inference:
        with st.spinner("Running inference..."):
            xgb_inference = run_xgb_inference(model_folder, demo_dfs)
        st.success("Inference completed successfully!")

    st.subheader("Step 3: View Results")
    if xgb_inference is not None:
        hourly_pred_h2_display = hourly_input_to_display(xgb_inference.hourly_pred_h2.iloc[0], pd.Series())
        hourly_pred_h2_display = hourly_pred_h2_display.dropna(axis=1, how='all')
        hourly_pred_h2_display.columns = ["Predicted 2nd Hour"]
        hourly_pred_h2_display["Observed 2nd Hour"] = hourly_display_df["2nd Hour"]
        hourly_pred_h3_display = hourly_input_to_display(xgb_inference.hourly_pred_h3_static.iloc[0], pd.Series())
        hourly_pred_h3_display = hourly_pred_h3_display.dropna(axis=1, how='all')
        hourly_pred_h3_display.columns = ["Static Predicted 3rd Hour"]
        hourly_pred_h3_dynamic_display = hourly_input_to_display(xgb_inference.hourly_pred_h3_dynamic.iloc[0], pd.Series())
        hourly_pred_h3_display["Dynamic Predicted 3rd Hour"] = hourly_pred_h3_dynamic_display["3rd Hour"]
        hourly_pred_h3_display["Observed 3rd Hour"] = hourly_display_df["3rd Hour"]
        st.dataframe(hourly_pred_h2_display)
        st.dataframe(hourly_pred_h3_display)


if __name__ == "__main__":
    main()
