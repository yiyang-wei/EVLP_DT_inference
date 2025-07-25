from inference.XGB_inference_new import *
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from huggingface_hub import snapshot_download


hourly_parameter_dict = {
    'pMean': 'Mean Airway Pressure (cmH₂O)',
    'PAP': 'Pulmonary Arterial Pressure (mmHg)',
    'Delta PO2': 'Delta PO₂ (mmHg)',
    'LA Ca++': 'Calcium (mmol/L)',
    'LA PCO2': 'Arterial Partial Pressure CO₂ (mmHg)',
    'LA HCO3': 'Bicarbonate (mmol/L)',
    'LAP': 'Left Atrial Pressure (mmHg)',
    'Calc Delta PCO2': 'Delta PCO₂ (mmHg)',
    'LA BE': 'Base Excess (mmol/L)',
    'LA K+': 'Potassium (mmol/L)',
    'pPlat': 'Plateau Airway Pressure (cmH₂O)',
    'LA CL': 'Chloride (mmol/L)',
    'Cstat': 'Static Compliance (mL/cmH₂O)',
    'Cdyn': 'Dynamic Compliance (mL/cmH₂O)',
    'PA PCO2': 'Venous Partial Pressure CO₂ (mmHg)',
    'LA Na+': 'Sodium (mmol/L)',
    'LA Lact': 'Lactate (mmol/L)',
    'LA Glu': 'Glucose (mmol/L)',
    'pPeak': 'Peak Airway Pressure (cmH₂O)',
    'LA pH': 'pH',
    'LA PO2': 'Arterial Partial Pressure O₂ (mmHg)',
    'PA PO2': 'Venous Partial Pressure O₂ (mmHg)',
    'STEEN lost': 'STEEN Lost (mL)',
}
reversed_hourly_parameter_dict = {v: k for k, v in hourly_parameter_dict.items()}

image_pc_dict = {
    'pca_0': 'Lung Xray PC1',
    'pca_1': 'Lung Xray PC2',
    'pca_2': 'Lung Xray PC3',
    'pca_3': 'Lung Xray PC4',
    'pca_4': 'Lung Xray PC5',
    'pca_5': 'Lung Xray PC6',
    'pca_6': 'Lung Xray PC7',
    'pca_7': 'Lung Xray PC8',
    'pca_8': 'Lung Xray PC9',
    'pca_9': 'Lung Xray PC10'
}

protein_dict = {
    'IL-6': 'Interleukin-6 (pg/mL)',
    'IL-8': 'Interleukin-8 (pg/mL)',
    'IL-10': 'Interleukin-10 (pg/mL)',
    'IL-1b': 'Interleukin-1β (pg/mL)'
}

gru_parameter_dict = {
    'Dy_comp(mL_cmH2O)': 'Dynamic Compliance (mL/cmH₂O)',
    'P_mean(cmH2O)': 'Mean Airway Pressure (cmH₂O)',
    'P_peak(cmH2O)': 'Peak Airway Pressure (cmH₂O)',
    'Ex_vol(mL)': 'Expiratory Volume (mL)'
}

@st.cache_data
def load_data(file_path, index_col=None):
    return pd.read_csv(file_path, index_col=index_col)

@st.cache_resource
def download_model(repo_id):
    model_path = snapshot_download(repo_id)
    return model_path

def hourly_input_to_display(hourly_input_one_case):
    hourly_features = []
    for feature in hourly_input_one_case.index:
        feature_without_prefix = feature.replace("70_", "").replace("80_", "").replace("90_", "")
        if feature_without_prefix not in hourly_features:
            hourly_features.append(feature_without_prefix)
    # hourly_features = [col.replace("70_", "") for col in
    #                    hourly_input_one_case.index[hourly_input_one_case.index.str.startswith("70_")]]
    hourly_prefixes = {"70_": "1st Hour", "80_": "2nd Hour", "90_": "3rd Hour"}
    hourly_display_one_case = pd.DataFrame(index=hourly_features, columns=[])
    for hour_prefix, hour_n in hourly_prefixes.items():
        hour_n_features = [hour_prefix + feature for feature in hourly_features if hour_prefix + feature in hourly_input_one_case.index]
        if not hour_n_features:
            continue
        hourly_display_one_case[hour_n] = hourly_input_one_case[hour_n_features].values
    hourly_display_one_case.rename(index=hourly_parameter_dict, inplace=True)
    return hourly_display_one_case

def hourly_display_to_input(hourly_display_one_case, case_id=0):
    hourly_input_one_case = pd.DataFrame(index=[case_id], columns=[])
    hourly_prefixes = {"70_": "1st Hour", "80_": "2nd Hour", "90_": "3rd Hour"}
    for feature in hourly_display_one_case.index:
        for hour_prefix, hour_n in hourly_prefixes.items():
            if feature in reversed_hourly_parameter_dict:
                colname = hour_prefix + reversed_hourly_parameter_dict[feature]
            else:
                colname = hour_prefix + feature
            hourly_input_one_case.loc[case_id, colname] = hourly_display_one_case.loc[feature, hour_n]
    return hourly_input_one_case

def image_pc_input_to_display(pc_1h_input_one_case, pc_3h_input_one_case):
    pc_features = [col.replace("feature_", "") for col in pc_1h_input_one_case.index]
    pc_case = pd.DataFrame(index=pc_features, columns=["1st Hour", "3rd Hour"])
    pc_case["1st Hour"] = pc_1h_input_one_case.values
    pc_case["3rd Hour"] = pc_3h_input_one_case.values
    pc_case.rename(index=image_pc_dict, inplace=True)
    return pc_case

def protein_input_to_display(protein_input_one_case):
    protein_features = list(protein_dict.keys())
    protein_prefixes = {"60_": "1st Hour", "90_": "90 Minutes", "110_": "110 Minutes",
                        "120_": "2nd Hour", "130_": "130 Minutes", "150_": "150 Minutes", "180_": "3rd Hour"}
    protein_display_one_case = pd.DataFrame(index=protein_features, columns=list(protein_prefixes.values()))
    for protein_prefix, protein_timestamp in protein_prefixes.items():
        protein_timestamp_features = [protein_prefix + feature for feature in protein_features]
        protein_display_one_case[protein_timestamp] = protein_input_one_case[protein_timestamp_features].values
    protein_display_one_case.rename(index=protein_dict, inplace=True)
    return protein_display_one_case

def calculate_protein_slopes(protein_input_one_case):
    protein_prefixes = {"60_": "1st Hour", "90_": "90 Minutes", "110_": "110 Minutes",
                        "120_": "2nd Hour", "130_": "130 Minutes", "150_": "150 Minutes", "180_": "3rd Hour"}
    protein_slope_case = pd.DataFrame(
        index=protein_input_one_case.index,
        columns=["Slope at 2nd Hour", "Slope at 150 Minutes", "Slope at 3rd Hour"])
    for feature in protein_input_one_case.index:
        slope_120_x = [60, 90, 110]
        slope_120, _ = np.polyfit(
            slope_120_x,
            protein_input_one_case.loc[feature, [protein_prefixes[f"{x}_"] for x in slope_120_x]].values,
            deg=1
        )
        protein_slope_case.loc[feature, "Slope at 2nd Hour"] = slope_120
        slope_150_x = [60, 90, 110, 120, 130]
        slope_150, _ = np.polyfit(
            slope_150_x,
            protein_input_one_case.loc[feature, [protein_prefixes[f"{x}_"] for x in slope_150_x]].values,
            deg=1
        )
        protein_slope_case.loc[feature, "Slope at 150 Minutes"] = slope_150
        slope_180_x = [60, 90, 110, 120, 130, 150]
        slope_180, _ = np.polyfit(
            slope_180_x,
            protein_input_one_case.loc[feature, [protein_prefixes[f"{x}_"] for x in slope_180_x]].values,
            deg=1
        )
        protein_slope_case.loc[feature, "Slope at 3rd Hour"] = slope_180
    return protein_slope_case

def transcriptomics_input_to_display(transcriptomics_input_x_one_case, transcriptomics_input_y_one_case):
    transcriptomics_features = [col.replace("_cit1", "") for col in transcriptomics_input_x_one_case.index]
    transcriptomics_case = pd.DataFrame(index=transcriptomics_features, columns=["Pre-perfusion", "Post-perfusion"])
    transcriptomics_case["Pre-perfusion"] = transcriptomics_input_x_one_case.values
    transcriptomics_case["Post-perfusion"] = transcriptomics_input_y_one_case[[f"{feature}_cit2" for feature in transcriptomics_features]].values
    return transcriptomics_case

def time_series_input_to_display(time_series_input_one_case: pd.DataFrame):
    breath_cols = time_series_input_one_case.columns[2:]
    ts_dfs = {
        "A1": pd.DataFrame(index=breath_cols, columns=list(gru_parameter_dict.values())),
        "A2": pd.DataFrame(index=breath_cols, columns=list(gru_parameter_dict.values())),
        "A3": pd.DataFrame(index=breath_cols, columns=list(gru_parameter_dict.values())),
    }
    for i, row in time_series_input_one_case.iterrows():
        sp = row["Simulated Parameter"]
        h = sp[:2]
        param = sp[6:]
        ts_dfs[h][gru_parameter_dict[param]] = row[breath_cols].values
    return ts_dfs


def main():

    st.set_page_config(
        page_title="EVLP DT Inference",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    st.title(":material/respiratory_rate: Ex-Vivo Lung Perfusion Digital Twin")

    data_folder = pathlib.Path("Data_for_demo")
    model_folder = pathlib.Path("Model")
    output_folder = pathlib.Path("Output")

    xgb_folder = model_folder / "XGB"
    hourly_model_folder = xgb_folder / "Hourly"
    protein_model_folder = xgb_folder / "Protein"
    transcriptomics_model_folder = xgb_folder / "Transcriptomics"

    # new_hourly_data_path = data_folder / "hourly_data_simulated.csv"
    # new_edema_data_path = data_folder / "edema_data_simulated.csv"
    # new_pc_1h_data_path = data_folder / "PC1h_data_simulated.csv"
    # new_pc_3h_data_path = data_folder / "PC3h_data_simulated.csv"
    # new_protein_data_path = data_folder / "protein_data_simulated_withslopes.csv"
    # new_transcriptomics_x_data_path = data_folder / "transcriptomics1_data_simulated.csv"
    # new_transcriptomics_y_data_path = data_folder / "transcriptomics2_data_simulated.csv"
    # new_time_series_data_path = data_folder / "ts_data_simulated.csv"
    new_hourly_data_path = data_folder / "Hourly.csv"
    new_edema_data_path = data_folder / "edema.csv"
    new_pc_1h_data_path = data_folder / "PC1h_data_simulated.csv"
    new_pc_3h_data_path = data_folder / "PC3h_data_simulated.csv"
    new_protein_data_path = data_folder / "Protein.csv"
    new_transcriptomics_x_data_path = data_folder / "microarray_cit1_data_simulated.csv"
    new_transcriptomics_y_data_path = data_folder / "microarray_cit2_data_simulated.csv"
    new_time_series_data_path = data_folder / "data_simulated.csv"

    # new_hourly_df = load_data(new_hourly_data_path, 1)
    # new_hourly_df.drop(columns=[new_hourly_df.columns[0]], inplace=True)
    # new_edema_df = load_data(new_edema_data_path, 1)
    # new_edema_df.drop(columns=[new_edema_df.columns[0]], inplace=True)
    # new_hourly_combined_df = pd.concat([new_hourly_df, new_edema_df], axis=1)
    #
    # new_pc_1h_df = load_data(new_pc_1h_data_path, 1)
    # new_pc_1h_df.drop(columns=[new_pc_1h_df.columns[0]], inplace=True)
    # new_pc_3h_df = load_data(new_pc_3h_data_path, 1)
    # new_pc_3h_df.drop(columns=[new_pc_3h_df.columns[0]], inplace=True)
    #
    # new_protein_data = load_data(new_protein_data_path, 1)
    # new_protein_data.drop(columns=[new_protein_data.columns[0]], inplace=True)
    #
    # new_transcriptomics_x_data = load_data(new_transcriptomics_x_data_path, 1)
    # new_transcriptomics_x_data.drop(columns=[new_transcriptomics_x_data.columns[0]], inplace=True)
    # new_transcriptomics_y_data = load_data(new_transcriptomics_y_data_path, 1)
    # new_transcriptomics_y_data.drop(columns=[new_transcriptomics_y_data.columns[0]], inplace=True)
    #
    # new_time_series_data = load_data(new_time_series_data_path)
    # new_time_series_data.drop(columns=[new_time_series_data.columns[0]], inplace=True)

    new_hourly_df = load_data(new_hourly_data_path, 0)
    new_edema_df = load_data(new_edema_data_path, 0)
    new_hourly_combined_df = pd.concat([new_hourly_df, new_edema_df], axis=1)

    new_pc_1h_df = load_data(new_pc_1h_data_path, 0)
    new_pc_3h_df = load_data(new_pc_3h_data_path, 0)

    new_protein_data = load_data(new_protein_data_path, 0)

    new_transcriptomics_x_data = load_data(new_transcriptomics_x_data_path, 0)
    new_transcriptomics_y_data = load_data(new_transcriptomics_y_data_path, 0)
    # print(f"Transcriptomics X data shape: {new_transcriptomics_x_data.shape}")
    # print(f"Transcriptomics Y data shape: {new_transcriptomics_y_data.shape}")
    # cit1_cols = (col.replace("_cit1", "") for col in new_transcriptomics_x_data.columns)
    # cit2_cols = (col.replace("_cit2", "") for col in new_transcriptomics_y_data.columns)
    # print(F"Missing columns in Transcriptomics X: {set(cit1_cols) - set(cit2_cols)}")
    # print(F"Missing columns in Transcriptomics Y: {set(cit2_cols) - set(cit1_cols)}")
    # print(len(new_transcriptomics_x_data.columns.unique()))
    # print(len(new_transcriptomics_y_data.columns.unique()))
    #
    # return

    new_time_series_data = load_data(new_time_series_data_path)

    with st.expander("Raw Input Tables", expanded=False):
        (
            raw_hourly_tab,
            raw_edema_tab,
            raw_pc_1h_tab,
            raw_pc_3h_tab,
            raw_protein_tab,
            raw_transcriptomics_x_tab,
            raw_transcriptomics_y_tab,
            raw_time_series_tab,
        ) = st.tabs([
            "Hourly Data",
            "Edema Data",
            "PC 1H Data",
            "PC 3H Data",
            "Protein Data",
            "Transcriptomics X Data",
            "Transcriptomics Y Data",
            "Time Series Data"
        ])
        with raw_hourly_tab:
            st.subheader("Hourly Data")
            st.dataframe(new_hourly_df, use_container_width=True)
        with raw_edema_tab:
            st.subheader("Edema Data")
            st.dataframe(new_edema_df, use_container_width=True)
        with raw_pc_1h_tab:
            st.subheader("PC 1H Data")
            st.dataframe(new_pc_1h_df, use_container_width=True)
        with raw_pc_3h_tab:
            st.subheader("PC 3H Data")
            st.dataframe(new_pc_3h_df, use_container_width=True)
        with raw_protein_tab:
            st.subheader("Protein Data")
            st.dataframe(new_protein_data, use_container_width=True)
        with raw_transcriptomics_x_tab:
            st.subheader("Transcriptomics X Data")
            st.dataframe(new_transcriptomics_x_data, use_container_width=True)
        with raw_transcriptomics_y_tab:
            st.subheader("Transcriptomics Y Data")
            st.dataframe(new_transcriptomics_y_data, use_container_width=True)
        with raw_time_series_tab:
            st.subheader("Time Series Data")
            st.dataframe(new_time_series_data, use_container_width=True)

    st.subheader("Step 1: Prepare Data")

    data_source = st.radio(
        "Use your own data or our demo data?",
        options=["Use Your Own Data", "Use Demo Data"],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        case_name_prefix = "Simulated Demo Case "
        selected_case_name = st.selectbox(
            "Select a case for inference",
            options=[case_name_prefix + str(i) for i in new_hourly_df.index],
            index=0,
        )
        selected_case_id = int(selected_case_name.replace(case_name_prefix, ""))
    with col2:
        inference_type = st.selectbox(
            "Select inference mode",
            options=["Static + Dynamic", "Static", "Dynamic"],
            index=0,
        )

    hourly_case = hourly_input_to_display(new_hourly_combined_df.loc[selected_case_id])

    pc_case = image_pc_input_to_display(new_pc_1h_df.loc[selected_case_id], new_pc_3h_df.loc[selected_case_id])

    protein_case = protein_input_to_display(new_protein_data.loc[selected_case_id])

    transcriptomics_case = transcriptomics_input_to_display(
        new_transcriptomics_x_data.loc[selected_case_id],
        new_transcriptomics_y_data.loc[selected_case_id]
    )

    time_series_case = time_series_input_to_display(new_time_series_data.loc[new_time_series_data["Demo ID"] == selected_case_id])

    with (st.expander(f"Data for {selected_case_name}", expanded=True)):
        (
            case_hourly_tab,
            case_pc_tab,
            case_protein_tab,
            case_transcriptomics_tab,
            case_time_series_a1_tab,
            case_time_series_a2_tab,
            case_time_series_a3_tab,
        ) = st.tabs([
            "Hourly Lung Function Data",
            "Lung Image Data",
            "Protein Data",
            "Transcriptomics Data",
            "1st Hour per-breath Data",
            "2nd Hour per-breath Data",
            "3rd Hour per-breath Data"
        ])
        with case_hourly_tab:
            hourly_case = st.data_editor(hourly_case, disabled=["_index"], use_container_width=True)
        with case_pc_tab:
            pc_case = st.data_editor(pc_case, disabled=["_index"], use_container_width=True)
        with case_protein_tab:
            protein_case = st.data_editor(protein_case, disabled=["_index"], use_container_width=True)
            protein_slope_case = calculate_protein_slopes(protein_case)
            st.dataframe(protein_slope_case, use_container_width=True)
        with case_transcriptomics_tab:
            transcriptomics_case = st.data_editor(transcriptomics_case, disabled=["_index"], use_container_width=True)
        with case_time_series_a1_tab:
            time_series_a1_case = st.data_editor(time_series_case["A1"], disabled=["_index"], use_container_width=True)
        with case_time_series_a2_tab:
            time_series_a2_case = st.data_editor(time_series_case["A2"], disabled=["_index"], use_container_width=True)
        with case_time_series_a3_tab:
            time_series_a3_case = st.data_editor(time_series_case["A3"], disabled=["_index"], use_container_width=True)

    # combine all the data into a single excel file
    excel_save_folder = pathlib.Path("Demo Data")
    excel_save_folder.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_save_folder / f"{selected_case_name}.xlsx") as writer:
        hourly_case.to_excel(writer, sheet_name='Hourly Lung Function Data', index=True)
        pc_case.to_excel(writer, sheet_name='Lung Image Data', index=True)
        protein_case.to_excel(writer, sheet_name='Protein Data', index=True)
        transcriptomics_case.to_excel(writer, sheet_name='Transcriptomics Data', index=True)
        time_series_a1_case.to_excel(writer, sheet_name='1st Hour per-breath Data', index=True)
        time_series_a2_case.to_excel(writer, sheet_name='2nd Hour per-breath Data', index=True)
        time_series_a3_case.to_excel(writer, sheet_name='3rd Hour per-breath Data', index=True)

    st.subheader("Run Inference")
    run_inference = st.button(
        label="Run Inference",
        icon=":material/play_arrow:",
        use_container_width=True
    )

    if not run_inference:
        return

    hourly_case_input = hourly_display_to_input(hourly_case)

    H1_to_H2_hourly_predicted_Y = predict_with_model(hourly_model_folder / "H1_to_H2", hourly_case_input)
    H1_to_H2_hourly_predicted_Y_display = hourly_input_to_display(H1_to_H2_hourly_predicted_Y.iloc[0])
    H1_to_H2_hourly_predicted_Y_display.columns = [f"Static Predicted {col}" for col in H1_to_H2_hourly_predicted_Y_display.columns]
    H1_H2_to_H3_hourly_predicted_Y = predict_with_model(hourly_model_folder / "H1_H2_to_H3", hourly_case_input)
    H1_H2_to_H3_hourly_predicted_Y_display = hourly_input_to_display(H1_H2_to_H3_hourly_predicted_Y.iloc[0])
    H1_H2_to_H3_hourly_predicted_Y_display.columns = [f"Dynamic Predicted {col}" for col in H1_H2_to_H3_hourly_predicted_Y_display.columns]

    H1_to_H2_H3_STEEN_predicted_Y = predict_with_model(hourly_model_folder / "STEEN_H1_to_H2_H3", hourly_case_input)
    H1_H2_to_H3_STEEN_predicted_Y = predict_with_model(hourly_model_folder / "STEEN_H1_H2_to_H3", hourly_case_input)

    hourly_h1 = hourly_case_input.loc[:, hourly_case_input.columns.str.startswith("70_")]
    hourly_h1_pred_h2 = pd.concat([hourly_h1, H1_to_H2_hourly_predicted_Y, H1_to_H2_H3_STEEN_predicted_Y], axis=1)
    H1_pred_H2_to_H3_hourly_predicted_Y = predict_with_model(hourly_model_folder / "H1_H2_to_H3", hourly_h1_pred_h2)
    H1_pred_H2_to_H3_STEEN_predicted_Y = predict_with_model(hourly_model_folder / "STEEN_H1_H2_to_H3", hourly_h1_pred_h2)

    H1_pred_H2_to_H3_hourly_predicted_Y_display = hourly_input_to_display(H1_pred_H2_to_H3_hourly_predicted_Y.iloc[0])
    H1_pred_H2_to_H3_hourly_predicted_Y_display.columns = [f"Static Predicted {col}" for col in H1_pred_H2_to_H3_hourly_predicted_Y_display.columns]

    hourly_inference_display = pd.concat([
        H1_to_H2_hourly_predicted_Y_display,
        H1_pred_H2_to_H3_hourly_predicted_Y_display,
        H1_H2_to_H3_hourly_predicted_Y_display,
    ], axis=1)
    hourly_inference_display.loc[hourly_parameter_dict["STEEN lost"], "Static Predicted 2nd Hour"] = \
        H1_to_H2_H3_STEEN_predicted_Y.loc[0, "80_STEEN lost"]
    hourly_inference_display.loc[hourly_parameter_dict["STEEN lost"], "Static Predicted 3rd Hour"] = \
        H1_pred_H2_to_H3_STEEN_predicted_Y.loc[0, "90_STEEN lost"]
    hourly_inference_display.loc[hourly_parameter_dict["STEEN lost"], "Dynamic Predicted 3rd Hour"] = \
        H1_H2_to_H3_STEEN_predicted_Y.loc[0, "90_STEEN lost"]

    (
        hourly_inference_tab,
        protein_inference_tab,
        transcriptomics_inference_tab,
    ) = st.tabs([
        "Hourly Inference Results",
        "Protein Inference Results",
        "Transcriptomics Inference Results"
    ])
    with hourly_inference_tab:
        st.dataframe(hourly_inference_display, use_container_width=True)
        st.download_button(
            label="Download Hourly Inference Results",
            data=hourly_inference_display.to_csv(index=True),
            file_name=f"{selected_case_name}_hourly_inference_results.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
