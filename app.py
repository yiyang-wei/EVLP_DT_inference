from inference.reformat import *
from inference.XGB_inference_new import XGBInference
from GRU.GRU import GRU
from inference.GRU_inference import TimeSeriesInference
from inference.visualization import *

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from huggingface_hub import snapshot_download
import time


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

def check_missing(data, tolerance=1.0):
    good = "✅ All Good"
    missing = "❌ {0:.1%} Missing"
    some_missing = "⚠️ {0:.1%} Missing"
    null_rate = data.isnull().to_numpy().flatten().mean()
    return good if null_rate == 0 else some_missing.format(null_rate) if null_rate < tolerance else missing.format(null_rate)

def check_modality_missing(dfs):
    hourly_1_status = check_missing(dfs[hourly_lung_function_sheet]["1st Hour"])
    hourly_2_status = check_missing(dfs[hourly_lung_function_sheet]["2nd Hour"])
    hourly_3_status = check_missing(dfs[hourly_lung_function_sheet]["3rd Hour"])
    pc_1_status = check_missing(dfs[lung_image_sheet]["1st Hour"])
    pc_3_status = check_missing(dfs[lung_image_sheet]["3rd Hour"])
    protein_1_status = check_missing(dfs[protein_sheet][["1st Hour", "90 Minutes", "110 Minutes"]])
    protein_2_status = check_missing(dfs[protein_sheet][["2nd Hour", "130 Minutes", "150 Minutes"]])
    protein_3_status = check_missing(dfs[protein_sheet]["3rd Hour"])
    cit1_status = check_missing(dfs[transcriptomics_sheet]["Baseline"])
    cit2_status = check_missing(dfs[transcriptomics_sheet]["Target"])
    a1_status = check_missing(dfs[per_breath_h1_sheet], tolerance=0)
    a2_status = check_missing(dfs[per_breath_h2_sheet], tolerance=0)
    a3_status = check_missing(dfs[per_breath_h3_sheet], tolerance=0)

    hourly_missing = pd.Series(name="Hourly Lung Function Data", index=["1st Hour", "2nd Hour"])
    hourly_missing["1st Hour"] = hourly_1_status
    hourly_missing["2nd Hour"] = hourly_2_status
    hourly_missing["3rd Hour"] = hourly_3_status
    lung_image_missing = pd.Series(name="Lung Image Data", index=["1st Hour", "3rd Hour"])
    lung_image_missing["1st Hour"] = pc_1_status
    lung_image_missing["3rd Hour"] = pc_3_status
    protein_missing = pd.Series(name="Protein Data", index=["1st Hour to 110 Minutes", "2nd Hour to 150 Minutes"])
    protein_missing["1st Hour to 110 Minutes"] = protein_1_status
    protein_missing["2nd Hour to 150 Minutes"] = protein_2_status
    protein_missing["3rd Hour"] = protein_3_status
    cit_missing = pd.Series(name="Transcriptomics Data", index=["Baseline", "Target"])
    cit_missing["Baseline"] = cit1_status
    cit_missing["Target"] = cit2_status
    ts_missing = pd.Series(name="Time-Series Data", index=["1Hr", "2Hr"])
    ts_missing["1Hr"] = a1_status
    ts_missing["2Hr"] = a2_status
    ts_missing["3Hr"] = a3_status

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.dataframe(hourly_missing)
    col2.dataframe(lung_image_missing)
    col3.dataframe(protein_missing)
    col4.dataframe(cit_missing)
    col5.dataframe(ts_missing)

    xgb_missing = any(status[0] != "✅" for status in (hourly_1_status, hourly_2_status, pc_1_status, pc_3_status, protein_1_status, protein_2_status, cit1_status))
    static_gru = a1_status[0] == "✅"
    dynamic_gru = static_gru and a2_status[0] == "✅"

    if xgb_missing:
        st.warning("DT will NOT be optimal due to missing data.")
    if not static_gru:
        st.warning("Static GRU inference will NOT be performed due to missing 1Hr per-breath data.")
    if not dynamic_gru:
        st.warning("Dynamic GRU inference will NOT be performed due to missing 2Hr per-breath data.")

    return static_gru, dynamic_gru

@st.cache_resource(show_spinner=False)
def run_xgb_inference(model_folder, demo_dfs):
    xgb_inference = XGBInference(model_folder)
    xgb_inference.load_input_data(demo_dfs)
    start = time.time()
    with st.spinner("Running Hourly Lung Function Inference..."):
        xgb_inference.hourly_dynamic_inference()
        xgb_inference.hourly_static_inference()
    end = time.time()
    st.success(f"Hourly Lung Function Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Lung Image Inference..."):
        xgb_inference.image_pc_dynamic_inference()
        xgb_inference.image_pc_static_inference()
    end = time.time()
    st.success(f"Lung Image Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Protein Inference..."):
        xgb_inference.protein_dynamic_inference()
        xgb_inference.protein_static_inference()
    end = time.time()
    st.success(f"Protein Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Transcriptomics Inference..."):
        xgb_inference.transcriptomics_dynamic_inference()
        xgb_inference.transcriptomics_static_inference()
        xgb_inference.get_pred_display()
    end = time.time()
    st.success(f"Transcriptomics Inference completed in {end - start:.2f} seconds.")
    return xgb_inference

@st.cache_resource(show_spinner=False)
def run_gru_inference(model_folder, demo_dfs, static_gru=True, dynamic_gru=True):
    time_series_inference = TimeSeriesInference(model_folder)
    time_series_inference.load_input_data(demo_dfs)
    if static_gru:
        start = time.time()
        with st.spinner("Running Static Time Series Inference"):
            time_series_inference.static_inference()
        end = time.time()
        st.success(f"Static Time Series Inference completed in {end - start:.2f} seconds.")
    if dynamic_gru:
        start = time.time()
        with st.spinner("Running Dynamic Time Series Inference"):
            time_series_inference.dynamic_inference()
        end = time.time()
        st.success(f"Dynamic Time Series Inference completed in {end - start:.2f} seconds.")
    return time_series_inference

def main():

    st.set_page_config(
        page_title="EVLP DT",
        page_icon=":material/respiratory_rate:",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "demo"

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

    col1, col2 = st.columns(2, border=True)

    with col1:
        st.button(
            "✔ " * (st.session_state["data_mode"] == "demo") + "Use Demo Data",
            type="primary" if st.session_state["data_mode"] == "demo" else "secondary",
            use_container_width=True,
            on_click=lambda: st.session_state.update(data_mode="demo"),
        )
        demo_files = data_folder.glob(demo_case_prefix + "*")
        demo_names = [file.stem for file in demo_files if file.is_file()]
        selected_demo_case = st.selectbox(
            label="Select a Demo Case",
            options=demo_names,
            index=0,
            disabled=st.session_state["data_mode"] != "demo",
        )

    with col2:
        st.button(
            "✔ " * (st.session_state["data_mode"] == "custom") + "Use Your Own Data",
            type="primary" if st.session_state["data_mode"] == "custom" else "secondary",
            use_container_width=True,
            on_click=lambda: st.session_state.update(data_mode="custom"),
        )

        st.download_button(
            label="Click Here to Download the Template (Excel File)",
            data=load_excel_binary(data_folder / "DT Lung Demo Template.xlsx"),
            type='tertiary',
            disabled=st.session_state["data_mode"] != "custom",
        )
        uploaded_case = st.file_uploader(
            label="Upload Your Own Data (must be an Excel file with the same format as the template)",
            type=["xlsx"],
            accept_multiple_files=False,
            disabled=st.session_state["data_mode"] != "custom",
        )

    if st.session_state["data_mode"] == "demo":
        case_dfs = load_excel(data_folder / f"{selected_demo_case}.xlsx")
        selected_case = selected_demo_case
    else:
        case_dfs = load_excel(uploaded_case) if uploaded_case else None
        selected_case = uploaded_case.name.replace(".xlsx", "") if uploaded_case else None

    if case_dfs is None:
        st.info("Upload an Excel to see the data preview and run inference.")
        return

    hourly_display_df = case_dfs[hourly_lung_function_sheet]
    image_pc_display_df = case_dfs[lung_image_sheet]
    protein_display_df = case_dfs[protein_sheet]
    transcriptomics_display_df = case_dfs[transcriptomics_sheet]
    time_series_display_dfs = [
        case_dfs[per_breath_h1_sheet],
        case_dfs[per_breath_h2_sheet],
        case_dfs[per_breath_h3_sheet]
    ]

    with (st.expander(f"Data Preview for {selected_case}", expanded=True)):
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

    static_gru, dynamic_gru = check_modality_missing(case_dfs)

    prediction_save_path = output_folder / f"{selected_case} predictions.xlsx"

    predictions_display = None
    if prediction_save_path.exists():
        st.info(f"Predictions for {selected_case} already exist. You can view the results below or re-run the inference.")
        predictions_display = load_excel(prediction_save_path)

    run_inference = st.button(
        label="Run Inference",
        icon=":material/play_arrow:",
        use_container_width=True
    )

    if run_inference:
        xgb_inference = run_xgb_inference(model_folder, case_dfs)
        gru_inference = run_gru_inference(model_folder, case_dfs, static_gru, dynamic_gru)
        st.success("✅ All Inference completed successfully!")
        with pd.ExcelWriter(prediction_save_path) as writer:
            for sheet_name, df in xgb_inference.predictions_display.items():
                df.to_excel(writer, sheet_name=sheet_name)
            gru_inference.pred_a2.to_excel(writer, sheet_name="2Hr Per-breath Prediction")
            gru_inference.static_pred_a3.to_excel(writer, sheet_name="3Hr Per-breath Static")
            gru_inference.dynamic_pred_a3.to_excel(writer, sheet_name="3Hr Per-breath Dynamic")
        predictions_display = load_excel(prediction_save_path)

    st.subheader("Step 3: View Results")
    if predictions_display is not None:
        with st.container(border=True):
            (
                hourly_pred_tab,
                image_pc_pred_tab,
                protein_pred_tab,
                transcriptomics_pred_tab,
                time_series_pred_tab,
            ) = st.tabs([
                "Hourly Lung Function Prediction",
                "Lung X-ray Image Prediction",
                "Protein Prediction",
                "Transcriptomics Prediction",
                "Per-breath Predictions",
            ])

        hourly_pred_tab.dataframe(predictions_display["Hourly Lung Function Prediction"])
        hourly_pred_tab.plotly_chart(
            hourly_all_features_line_plot(predictions_display["Hourly Lung Function Prediction"]),
            use_container_width=True,
        )

        image_pc_pred_tab.dataframe(predictions_display["Lung X-ray Image Prediction"])
        image_pc_pred_tab.plotly_chart(
            image_pc_line_plot(predictions_display["Lung X-ray Image Prediction"]),
            use_container_width=True,
        )

        protein_pred_tab.dataframe(predictions_display["Protein Prediction"])
        protein_pred_tab.plotly_chart(
            protein_line_plot(predictions_display["Protein Prediction"]),
            use_container_width=True,
            key="protein_line_plot_1"
        )
        protein_pred_tab.plotly_chart(
            protein_line_plot_2(predictions_display["Protein Prediction"]),
            use_container_width=True,
            key="protein_line_plot_2"
        )

        transcriptomics_pred_tab.dataframe(predictions_display["Transcriptomics Prediction"])
        transcriptomics_pred_tab.plotly_chart(
            transcriptomics_heatmap(predictions_display["Transcriptomics Prediction"]),
            use_container_width=True,
        )
        transcriptomics_pred_tab.plotly_chart(
            transcriptomics_bar_plot(predictions_display["Transcriptomics Prediction"]),
            use_container_width=True,
        )

        time_series_pred_tab.markdown("**2Hr Per-breath Prediction**")
        time_series_pred_tab.dataframe(predictions_display["2Hr Per-breath Prediction"])
        col1, col2 = time_series_pred_tab.columns(2)
        col1.markdown("**3Hr Per-breath Static Prediction**")
        col1.dataframe(predictions_display["3Hr Per-breath Static"])
        col2.markdown("**3Hr Per-breath Dynamic Prediction**")
        col2.dataframe(predictions_display["3Hr Per-breath Dynamic"])

        figs = timeseries_plot(
            case_dfs[per_breath_h1_sheet],
            case_dfs[per_breath_h2_sheet],
            case_dfs[per_breath_h3_sheet],
            predictions_display["2Hr Per-breath Prediction"],
            predictions_display["3Hr Per-breath Static"],
            predictions_display["3Hr Per-breath Dynamic"]
        )
        col1, col2 = time_series_pred_tab.columns(2)
        col1.plotly_chart(figs[0], use_container_width=True)
        col1.plotly_chart(figs[1], use_container_width=True)
        col2.plotly_chart(figs[2], use_container_width=True)
        col2.plotly_chart(figs[3], use_container_width=True)

    else:
        st.warning("No predictions available to view. Please run the inference first.")

    st.subheader("Step 4: Download Predictions")
    if prediction_save_path.exists():
        saved_predictions = load_excel_binary(prediction_save_path)

        st.download_button(
            label="Download Predictions",
            data=saved_predictions,
            file_name=f"{selected_case} predictions.xlsx",
            mime="application/vnd.ms-excel",
            disabled=saved_predictions is None,
            use_container_width=True
        )
    else:
        st.warning("No predictions available to download. Please run the inference first.")



if __name__ == "__main__":
    main()
