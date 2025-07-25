import streamlit as st
import numpy as np
import pandas as pd
import pathlib
# import openpyxl


hourly_order = {
    "70": "1st Hour",
    "80": "2nd Hour",
    "90": "3rd Hour",
}
hourly_order_reversed = {v: k for k, v in hourly_order.items()}
hourly_name = {
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
hourly_code = {v: k for k, v in hourly_name.items()}
def get_hourly_display_name(colname):
    hour_prefix, feature_code = colname.split("_", 1)
    feature_name = hourly_name.get(feature_code, feature_code)
    timestamp_name = hourly_order.get(hour_prefix, hour_prefix)
    return feature_name, timestamp_name
def get_hourly_input_name(display_name, timestamp):
    if display_name in hourly_code:
        feature_code = hourly_code[display_name]
    else:
        raise KeyError(f"Display name '{display_name}' not found in hourly display name to input name mapping.")
    if timestamp in hourly_order_reversed:
        hour_prefix = hourly_order_reversed[timestamp]
    else:
        raise KeyError(f"Timestamp '{timestamp}' not found in hourly order mapping.")
    return f"{hour_prefix}_{feature_code}"

image_pc_order = {
    "_x": "1st Hour",
    "_y": "3rd Hour"
}
image_pc_order_reversed = {v: k for k, v in image_pc_order.items()}
image_pc_name = {
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
image_pc_code = {v: k for k, v in image_pc_name.items()}
def get_image_pc_display_name(colname):
    feature_code = colname.replace("feature_", "")
    feature_name = image_pc_name.get(feature_code, feature_code)
    return feature_name
def get_image_pc_input_name(display_name):
    if display_name in image_pc_code:
        feature_code = image_pc_code[display_name]
    else:
        raise KeyError(f"Display name '{display_name}' not found in image PC display name to input name mapping.")
    return f"feature_{feature_code}"

protein_order = {
    60: "1st Hour",
    90: "90 Minutes",
    110: "110 Minutes",
    120: "2nd Hour",
    130: "130 Minutes",
    150: "150 Minutes",
    180: "3rd Hour"
}
protein_order_reversed = {v: k for k, v in protein_order.items()}
protein_name = {
    'IL-6': 'Interleukin-6 (pg/mL)',
    'IL-8': 'Interleukin-8 (pg/mL)',
    'IL-10': 'Interleukin-10 (pg/mL)',
    'IL-1b': 'Interleukin-1β (pg/mL)'
}
protein_slope_order = {
    'slope_120': 'Slope at 2nd Hour',
    'slope_150': 'Slope at 150 Minutes',
    'slope_180': 'Slope at 3rd Hour'
}
protein_code = {v: k for k, v in protein_name.items()}
protein_slope_order_reversed = {v: k for k, v in protein_slope_order.items()}
def get_protein_display_name(colname):
    protein_prefix, feature_code = colname.split("_", 1)
    feature_name = protein_name.get(feature_code, feature_code)
    timestamp_name = protein_order.get(int(protein_prefix), protein_prefix)
    return feature_name, timestamp_name
def get_protein_input_name(display_name, timestamp):
    if display_name in protein_code:
        feature_code = protein_code[display_name]
    else:
        raise KeyError(f"Display name '{display_name}' not found in protein display name to input name mapping.")
    if timestamp in protein_order_reversed:
        protein_prefix = protein_order_reversed[timestamp]
    else:
        raise KeyError(f"Timestamp '{timestamp}' not found in protein order mapping.")
    return f"{protein_prefix}_{feature_code}"
def get_protein_slope_input_name(display_name, timestamp):
    if display_name in protein_code:
        feature_code = protein_code[display_name]
    else:
        raise KeyError(f"Display name '{display_name}' not found in protein display name to input name mapping.")
    if timestamp in protein_slope_order_reversed:
        slope_sufix = protein_slope_order_reversed[timestamp]
    else:
        raise KeyError(f"Timestamp '{timestamp}' not found in protein order mapping.")
    return f"{feature_code}_{slope_sufix}"

transcriptomics_order = {
    "_cit1": "Pre-perfusion",
    "_cit2": "Post-perfusion"
}
transcriptomics_order_reversed = {v: k for k, v in transcriptomics_order.items()}

def get_transcriptomics_display_name(colname):
    feature_name = colname.replace("_cit1", "").replace("_cit2", "")
    cit = colname[-5:]
    timestamp_name = transcriptomics_order.get(cit)
    return feature_name, timestamp_name

def get_transcriptomics_input_name(display_name, timestamp):
    if timestamp in transcriptomics_order_reversed:
        cit = transcriptomics_order_reversed[timestamp]
    else:
        raise KeyError(f"Timestamp '{timestamp}' not found in transcriptomics order mapping.")
    return f"{display_name}{cit}"

per_breath_name = {
    'Dy_comp(mL_cmH2O)': 'Dynamic Compliance (mL/cmH₂O)',
    'P_mean(cmH2O)': 'Mean Airway Pressure (cmH₂O)',
    'P_peak(cmH2O)': 'Peak Airway Pressure (cmH₂O)',
    'Ex_vol(mL)': 'Expiratory Volume (mL)'
}
per_breath_code = {v: k for k, v in per_breath_name.items()}

hourly_lung_function_sheet = "Hourly Lung Function Data"
lung_image_sheet = "Lung Image Data"
protein_sheet = "Protein Data"
transcriptomics_sheet = "Transcriptomics Data"
per_breath_h1_sheet = "1st Hour per-breath Data"
per_breath_h2_sheet = "2nd Hour per-breath Data"
per_breath_h3_sheet = "3rd Hour per-breath Data"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def choose_id_column(data, key):
    id_col = st.selectbox(
        label=f"Choose Case ID column",
        options=list(data.columns),
        index=0,
        key=f"id_col_{key}"
    )
    data.set_index(id_col, inplace=True, drop=True)

def choose_id_column_timeseries(data, key):
    id_col = st.selectbox(
        label=f"Choose Case ID column",
        options=list(data.columns),
        index=0,
        key=f"id_col_{key}"
    )
    param_col = st.selectbox(
        label=f"Choose Parameter column",
        options=list(data.columns),
        index=1,
        key=f"param_col_{key}"
    )
    data.set_index([id_col, param_col], inplace=True, drop=True)

def preview_data(data):
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Shape", value=str(data.shape), border=True)
    col2.metric(label="Missing", value=data.isnull().sum().sum(), border=True)
    col3.metric(label="Memory Usage", value=f"{data.memory_usage(deep=True).sum() / 1024:.2f} KB", border=True)
    st.dataframe(data, use_container_width=True)

def get_csv_input(uploader_label, key, ts=False):
    with st.container(border=True):
        uploaded_file = st.file_uploader(label=uploader_label, type=["csv"], key=f"uploader {key}")
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            columns = list(data.columns)
            with st.expander("Columns to use", expanded=False):
                cols_to_use = st.pills(
                    label="Select columns to use",
                    label_visibility="collapsed",
                    selection_mode="multi",
                    options=columns,
                    default=columns,
                    key=f"cols_to_use {key}"
                )
                data = data[cols_to_use]
            if ts:
                choose_id_column_timeseries(data, key)
            else:
                choose_id_column(data, key)
            preview_data(data)
            return data
        else:
            return None

def hourly_input_to_display(hourly_input_one_case, emeda_input_one_case):
    combined_input_one_case = pd.concat([hourly_input_one_case, emeda_input_one_case])
    features = list(hourly_name.values())
    prefixes = list(hourly_order.values())
    hourly_display_one_case = pd.DataFrame(index=features, columns=prefixes)
    for code in combined_input_one_case.index:
        name, timestamp = get_hourly_display_name(code)
        hourly_display_one_case.loc[name, timestamp] = combined_input_one_case[code]
    return hourly_display_one_case

def hourly_display_to_input(hourly_display_one_case):
    hourly_input_one_case = pd.Series()
    for name in hourly_display_one_case.index:
        for timestamp in hourly_display_one_case.columns:
            colname = get_hourly_input_name(name, timestamp)
            hourly_input_one_case[colname] = hourly_display_one_case.loc[name, timestamp]
    return hourly_input_one_case

def image_pc_input_to_display(pc_1h_input_one_case, pc_3h_input_one_case):
    features = list(image_pc_name.values())
    timestamps = list(image_pc_order.values())
    pc_case = pd.DataFrame(index=features, columns=timestamps)
    pc_case[timestamps[0]] = pc_1h_input_one_case[[get_image_pc_input_name(name) for name in features]].values
    pc_case[timestamps[1]] = pc_3h_input_one_case[[get_image_pc_input_name(name) for name in features]].values
    return pc_case

def image_pc_display_to_input(pc_display_one_case):
    pc_h1_input_one_case = pd.Series()
    pc_h3_input_one_case = pd.Series()
    timestamps = list(image_pc_order.values())
    for name in pc_display_one_case.index:
        colname = get_image_pc_input_name(name)
        pc_h1_input_one_case[colname] = pc_display_one_case.loc[name, timestamps[0]]
        pc_h3_input_one_case[colname] = pc_display_one_case.loc[name, timestamps[1]]
    return pc_h1_input_one_case, pc_h3_input_one_case

def protein_input_to_display(protein_input_one_case):
    features = list(protein_name.values())
    timestamps = list(protein_order.values())
    protein_case = pd.DataFrame(index=features, columns=timestamps)
    for code in protein_input_one_case.index:
        if "slope" in code:
            continue
        name, timestamp = get_protein_display_name(code)
        protein_case.loc[name, timestamp] = protein_input_one_case[code]
    return protein_case

def calculate_protein_slopes(protein_input_one_case):
    protein_slope_case = pd.DataFrame(
        index=protein_input_one_case.index,
        columns=list(protein_slope_order.values()))
    timestamps = list(protein_order.keys())
    for feature in protein_input_one_case.index:
        for slope_code, slope_name in protein_slope_order.items():
            slope_timestamp = int(slope_code.split("_")[1])
            slope_x = [t for t in timestamps if t < slope_timestamp]
            slope_y = protein_input_one_case.loc[feature, [protein_order[t] for t in slope_x]].to_list()
            slope, _ = np.polyfit(slope_x, slope_y, deg=1)
            protein_slope_case.loc[feature, slope_name] = slope
    return protein_slope_case

def protein_display_to_input(protein_display_one_case):
    protein_input_one_case = pd.Series()
    for name in protein_display_one_case.index:
        for timestamp in protein_display_one_case.columns:
            colname = get_protein_input_name(name, timestamp)
            protein_input_one_case[colname] = protein_display_one_case.loc[name, timestamp]
    return protein_input_one_case

def protein_slope_display_to_input(protein_slope_display_one_case):
    protein_slope_input_one_case = pd.Series()
    for name in protein_slope_display_one_case.index:
        for timestamp in protein_slope_display_one_case.columns:
            colname = get_protein_slope_input_name(name, timestamp)
            protein_slope_input_one_case[colname] = protein_slope_display_one_case.loc[name, timestamp]
    return protein_slope_input_one_case

def transcriptomics_input_to_display(transcriptomics_input_x_one_case, transcriptomics_input_y_one_case):
    combined_transcriptomics = pd.concat([transcriptomics_input_x_one_case, transcriptomics_input_y_one_case])
    transcriptomics_case = pd.DataFrame()
    for col in combined_transcriptomics.index:
        feature_name, timestamp = get_transcriptomics_display_name(col)
        transcriptomics_case.loc[feature_name, timestamp] = combined_transcriptomics[col]
    return transcriptomics_case

def transcriptomics_display_to_input(transcriptomics_display_one_case):
    transcriptomics_input_one_case = pd.Series()
    for name in transcriptomics_display_one_case.index:
        for timestamp in transcriptomics_display_one_case.columns:
            colname = get_transcriptomics_input_name(name, timestamp)
            transcriptomics_input_one_case[colname] = transcriptomics_display_one_case.loc[name, timestamp]
    return transcriptomics_input_one_case


def time_series_input_to_display(time_series_input_one_case: pd.DataFrame):
    breath_cols = time_series_input_one_case.columns
    ts_dfs = {
        "A1": pd.DataFrame(index=breath_cols, columns=list(per_breath_name.values())),
        "A2": pd.DataFrame(index=breath_cols, columns=list(per_breath_name.values())),
        "A3": pd.DataFrame(index=breath_cols, columns=list(per_breath_name.values())),
    }
    for i, row in time_series_input_one_case.iterrows():
        sp = row.name
        h = sp[:2]
        param = sp[6:]
        ts_dfs[h][per_breath_name[param]] = row[breath_cols].values
    return ts_dfs


def main():

    st.set_page_config(
        page_title="EVLP DT Data Reformatting Tool",
        page_icon="🧰",
        layout="wide"
    )

    st.title("🧰EVLP DT Data Reformatting Tool")

    st.subheader("Step 1: Upload Input Data")

    (
        hourly_input_tab,
        edema_input_tab,
        pc_1hr_input_tab,
        pc_3hr_input_tab,
        protein_input_tab,
        cit1_input_tab,
        cit2_input_tab2,
        ts_input_tab,
    ) = st.tabs([
        "Hourly data",
        "Edema data",
        "Image PC 1hr data",
        "Image PC 3hr data",
        "Protein data",
        "Transcriptomics cit1 data",
        "Transcriptomics cit2 data",
        "Timeseries data"
    ])

    with hourly_input_tab:
        hourly_input_df = get_csv_input(
            uploader_label=f"Upload Hourly data CSV file",
            key="hourly_input",
        )
    with edema_input_tab:
        edema_input_df = get_csv_input(
            uploader_label=f"Upload Edema data CSV file",
            key="edema_input",
        )
    with pc_1hr_input_tab:
        pc_1hr_input_df = get_csv_input(
            uploader_label=f"Upload Image PC 1hr data CSV file",
            key="pc_1hr_input",
        )
    with pc_3hr_input_tab:
        pc_3hr_input_df = get_csv_input(
            uploader_label=f"Upload Image PC 3hr data CSV file",
            key="pc_3hr_input",
        )
    with protein_input_tab:
        protein_input_df = get_csv_input(
            uploader_label=f"Upload Protein data CSV file",
            key="protein_input",
        )
    with cit1_input_tab:
        cit1_input_df = get_csv_input(
            uploader_label=f"Upload Transcriptomics cit1 data CSV file",
            key="cit1_input",
        )
    with cit2_input_tab2:
        cit2_input_df = get_csv_input(
            uploader_label=f"Upload Transcriptomics cit2 data CSV file",
            key="cit2_input",
        )
    with ts_input_tab:
        timeseries_input_name = "Timeseries data"
        timeseries_input_df = get_csv_input(
            uploader_label=f"Upload Timeseries data CSV file",
            key="ts_input",
            ts=True
        )

    cases = []
    if hourly_input_df is not None:
        cases = hourly_input_df.index
    elif edema_input_df is not None:
        cases = edema_input_df.index
    elif pc_1hr_input_df is not None:
        cases = pc_1hr_input_df.index
    elif pc_3hr_input_df is not None:
        cases = pc_3hr_input_df.index
    elif protein_input_df is not None:
        cases = protein_input_df.index
    elif cit1_input_df is not None:
        cases = cit1_input_df.index
    elif cit2_input_df is not None:
        cases = cit2_input_df.index
    elif timeseries_input_df is not None:
        cases = timeseries_input_df.index.get_level_values(0).unique()

    st.subheader("Step 2: Review Display Data")
    case_name_prefix = "Simulated Demo Case "
    selected_case_name = st.selectbox(
        "Select a case for display",
        options=(case_name_prefix + str(case) for case in cases),
        index=0,
    )
    if selected_case_name is not None:
        selected_case_id = int(selected_case_name.replace(case_name_prefix, ""))
    (
        hourly_display_tab,
        pc_display_tab,
        protein_display_tab,
        transcriptomics_display_tab,
        ts_h1_display_tab,
        ts_h2_display_tab,
        ts_h3_display_tab,
    ) = st.tabs([
        hourly_lung_function_sheet,
        lung_image_sheet,
        protein_sheet,
        transcriptomics_sheet,
        per_breath_h1_sheet,
        per_breath_h2_sheet,
        per_breath_h3_sheet
    ])

    hourly_display_df = None
    pc_display_df = None
    protein_display_df = None
    transcriptomics_display_df = None

    if hourly_input_df is None:
        hourly_display_tab.warning("No Hourly data uploaded.")
    if edema_input_df is None:
        hourly_display_tab.warning("No Edema data uploaded.")
    if hourly_input_df is not None and edema_input_df is not None:
        hourly_display_df = hourly_input_to_display(hourly_input_df.loc[selected_case_id], edema_input_df.loc[selected_case_id])
        hourly_display_tab.dataframe(hourly_display_df, use_container_width=True)

    if pc_1hr_input_df is None:
        pc_display_tab.warning("No Image PC 1hr data uploaded.")
    if pc_3hr_input_df is None:
        pc_display_tab.warning("No Image PC 3hr data uploaded.")
    if pc_1hr_input_df is not None and pc_3hr_input_df is not None:
        pc_display_df = image_pc_input_to_display(pc_1hr_input_df.loc[selected_case_id], pc_3hr_input_df.loc[selected_case_id])
        pc_display_tab.dataframe(pc_display_df, use_container_width=True)

    if protein_input_df is None:
        protein_display_tab.warning("No Protein data uploaded.")
    if protein_input_df is not None:
        protein_display_df = protein_input_to_display(protein_input_df.loc[selected_case_id])
        protein_display_tab.dataframe(protein_display_df, use_container_width=True)
        protein_slope_df = calculate_protein_slopes(protein_display_df)
        protein_display_tab.dataframe(protein_slope_df, use_container_width=True)

    if cit1_input_df is None:
        transcriptomics_display_tab.warning("No Transcriptomics cit1 data uploaded.")
    if cit2_input_df is None:
        transcriptomics_display_tab.warning("No Transcriptomics cit2 data uploaded.")
    if cit1_input_df is not None and cit2_input_df is not None:
        transcriptomics_display_df = transcriptomics_input_to_display(
            cit1_input_df.loc[selected_case_id],
            cit2_input_df.loc[selected_case_id]
        )
        transcriptomics_display_tab.dataframe(transcriptomics_display_df, use_container_width=True)

    if timeseries_input_df is None:
        ts_h1_display_tab.warning("No Timeseries data uploaded.")
        ts_h2_display_tab.warning("No Timeseries data uploaded.")
        ts_h3_display_tab.warning("No Timeseries data uploaded.")
    else:
        timeseries_display_dfs = time_series_input_to_display(timeseries_input_df.loc[selected_case_id])
        ts_h1_display_tab.dataframe(timeseries_display_dfs["A1"], use_container_width=True)
        ts_h2_display_tab.dataframe(timeseries_display_dfs["A2"], use_container_width=True)
        ts_h3_display_tab.dataframe(timeseries_display_dfs["A3"], use_container_width=True)

    st.subheader("Step 3: Convert to Model Input")
    (
        hourly_model_input_tab,
        pc_model_input_tab,
        protein_model_input_tab,
        cit_model_input_tab,
    ) = st.tabs([
        "Hourly data",
        "Image PC data",
        "Protein data",
        "Transcriptomics data",
    ])
    if hourly_display_df is not None:
        hourly_model_input_df = pd.DataFrame([hourly_display_to_input(hourly_display_df)], index=[selected_case_id])
        hourly_model_input_tab.dataframe(hourly_model_input_df)
        if np.allclose(hourly_model_input_df.loc[selected_case_id, hourly_input_df.columns], hourly_input_df.loc[selected_case_id]):
            hourly_model_input_tab.success("Reverted hourly data matches the original input data.")
        else:
            hourly_model_input_tab.error("Reverted Hourly data does not match the original input data.")
        if np.allclose(hourly_model_input_df.loc[selected_case_id, edema_input_df.columns], edema_input_df.loc[selected_case_id]):
            hourly_model_input_tab.success("Reverted Edema data matches the original input data.")
        else:
            hourly_model_input_tab.error("Reverted Edema data does not match the original input data.")
    if pc_display_df is not None:
        pc_model_input_h1_df, pc_model_input_h3_df = image_pc_display_to_input(pc_display_df)
        pc_model_input_h1_df = pd.DataFrame([pc_model_input_h1_df], index=[selected_case_id])
        pc_model_input_h3_df = pd.DataFrame([pc_model_input_h3_df], index=[selected_case_id])
        pc_model_input_tab.dataframe(pc_model_input_h1_df, use_container_width=True)
        pc_model_input_tab.dataframe(pc_model_input_h3_df, use_container_width=True)
        if np.allclose(pc_model_input_h1_df.loc[selected_case_id, pc_1hr_input_df.columns], pc_1hr_input_df.loc[selected_case_id]):
            pc_model_input_tab.success("Reverted Image PC 1hr data matches the original input data.")
        else:
            pc_model_input_tab.error("Reverted Image PC 1hr data does not match the original input data.")
        if np.allclose(pc_model_input_h3_df.loc[selected_case_id, pc_3hr_input_df.columns], pc_3hr_input_df.loc[selected_case_id]):
            pc_model_input_tab.success("Reverted Image PC 3hr data matches the original input data.")
        else:
            pc_model_input_tab.error("Reverted Image PC 3hr data does not match the original input data.")

    if protein_display_df is not None:
        protein_model_input_df = pd.DataFrame([protein_display_to_input(protein_display_df)], index=[selected_case_id])
        protein_slope_input_df = pd.DataFrame([protein_slope_display_to_input(protein_slope_df)], index=[selected_case_id])
        protein_with_slope_model_input_df = pd.concat([protein_model_input_df, protein_slope_input_df], axis=1)
        protein_model_input_tab.dataframe(protein_with_slope_model_input_df, use_container_width=True)
        if np.allclose(protein_with_slope_model_input_df.loc[selected_case_id, protein_input_df.columns], protein_input_df.loc[selected_case_id]):
            protein_model_input_tab.success("Reverted Protein data matches the original input data.")
        else:
            protein_model_input_tab.error("Reverted Protein data does not match the original input data.")

    if transcriptomics_display_df is not None:
        transcriptomics_model_input_df = pd.DataFrame([transcriptomics_display_to_input(transcriptomics_display_df)], index=[selected_case_id])
        cit_model_input_tab.dataframe(transcriptomics_model_input_df, use_container_width=True)
        if np.allclose(transcriptomics_model_input_df.loc[selected_case_id, cit1_input_df.columns], cit1_input_df.loc[selected_case_id]):
            cit_model_input_tab.success("Reverted Transcriptomics cit1 data matches the original input data.")
        else:
            cit_model_input_tab.error("Reverted Transcriptomics cit1 data does not match the original input data.")
        if np.allclose(transcriptomics_model_input_df.loc[selected_case_id, cit2_input_df.columns], cit2_input_df.loc[selected_case_id]):
            cit_model_input_tab.success("Reverted Transcriptomics cit2 data matches the original input data.")
        else:
            cit_model_input_tab.error("Reverted Transcriptomics cit2 data does not match the original input data.")

    st.subheader("Step 4: Download Display Data")
    all_exist = True
    if hourly_input_df is None:
        st.warning("No hourly input data uploaded.")
        all_exist = False
    if edema_input_df is None:
        st.warning("No edema input data uploaded.")
        all_exist = False
    if pc_1hr_input_df is None:
        st.warning("No Image PC 1hr input data uploaded.")
        all_exist = False
    if pc_3hr_input_df is None:
        st.warning("No Image PC 3hr input data uploaded.")
        all_exist = False
    if protein_input_df is None:
        st.warning("No Protein input data uploaded.")
        all_exist = False
    if cit1_input_df is None:
        st.warning("No Transcriptomics cit1 input data uploaded.")
        all_exist = False
    if cit2_input_df is None:
        st.warning("No Transcriptomics cit2 input data uploaded.")
        all_exist = False

    col1, col2 = st.columns(2, vertical_alignment='bottom')

    save_folder_name = col1.text_input(
        label="Folder name to save Excel files",
        value="Display Data",
        placeholder="Enter folder name"
    )

    download_all_display_button = col2.button(
        label="Download per-case Excel for all cases",
        disabled=not all_exist or save_folder_name is None,
        use_container_width=True
    )

    if download_all_display_button:
        save_folder = pathlib.Path(save_folder_name)
        save_folder.mkdir(parents=True, exist_ok=True)
        for case in cases:
            case_name = f"Simulated Demo Case {case}"
            with pd.ExcelWriter(save_folder / f"{case_name}.xlsx") as writer:
                hourly_display_df = hourly_input_to_display(
                    hourly_input_df.loc[case],
                    edema_input_df.loc[case]
                )
                pc_display_df = image_pc_input_to_display(
                    pc_1hr_input_df.loc[case],
                    pc_3hr_input_df.loc[case]
                )
                protein_display_df = protein_input_to_display(protein_input_df.loc[case])
                transcriptomics_display_df = transcriptomics_input_to_display(
                    cit1_input_df.loc[case],
                    cit2_input_df.loc[case]
                )
                timeseries_display_dfs = time_series_input_to_display(timeseries_input_df.loc[case])
                hourly_display_df.to_excel(writer, sheet_name=hourly_lung_function_sheet)
                pc_display_df.to_excel(writer, sheet_name=lung_image_sheet)
                protein_display_df.to_excel(writer, sheet_name=protein_sheet)
                transcriptomics_display_df.to_excel(writer, sheet_name=transcriptomics_sheet)
                timeseries_display_dfs["A1"].to_excel(writer, sheet_name=per_breath_h1_sheet)
                timeseries_display_dfs["A2"].to_excel(writer, sheet_name=per_breath_h2_sheet)
                timeseries_display_dfs["A3"].to_excel(writer, sheet_name=per_breath_h3_sheet)
        st.success(f"All display data saved in {save_folder_name} folder.")


if __name__ == '__main__':
    main()


