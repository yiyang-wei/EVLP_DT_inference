import numpy as np
import pandas as pd

hourly_lung_function_sheet = "Lung Function Hourly Data"
lung_image_sheet = "Lung X-ray Image Data"
protein_sheet = "Protein Data"
transcriptomics_sheet = "Transcriptomics Data"
per_breath_h1_sheet = "1Hr Per-breath Data"
per_breath_h2_sheet = "2Hr Per-breath Data"
per_breath_h3_sheet = "3Hr Per-breath Data"


hourly_order = {
    "70": "1st Hour",
    "80": "2nd Hour",
    "90": "3rd Hour",
}
hourly_order_reversed = {v: k for k, v in hourly_order.items()}
hourly_name = {
    'pMean': 'Mean Airway Pressure (cmH₂O)',
    'PAP': 'Pulmonary Arterial Pressure (mmHg)',
    'LA Ca++': 'Calcium (mmol/L)',
    'LA PCO2': 'Arterial Partial Pressure CO₂ (mmHg)',
    'LA HCO3': 'Bicarbonate (mmol/L)',
    'LAP': 'Left Atrial Pressure (mmHg)',
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
    'STEEN lost': 'Edema (mL)',
}
hourly_calculated_name = {
    'Delta PO2': 'Delta PO₂ (mmHg)',
    'Calc Delta PCO2': 'Delta PCO₂ (mmHg)',
}
hourly_code = {v: k for k, v in hourly_name.items()}
hourly_calculated_code = {v: k for k, v in hourly_calculated_name.items()}
def get_hourly_display_name(colname):
    hour_prefix, feature_code = colname.split("_", 1)
    feature_name = hourly_name.get(feature_code)
    timestamp_name = hourly_order.get(hour_prefix)
    return feature_name, timestamp_name
def get_hourly_input_name(display_name, timestamp):
    if display_name in hourly_code:
        feature_code = hourly_code[display_name]
    elif display_name in hourly_calculated_code:
        feature_code = hourly_calculated_code[display_name]
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
    feature_name = image_pc_name.get(feature_code)
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
    'slope_120': 'Derived slope at 2Hr',
    'slope_150': 'Derived slope at 150Mins',
    'slope_180': 'Derived slope at 3Hr'
}
protein_code = {v: k for k, v in protein_name.items()}
protein_slope_order_reversed = {v: k for k, v in protein_slope_order.items()}
def get_protein_display_name(colname):
    protein_prefix, feature_code = colname.split("_", 1)
    feature_name = protein_name.get(feature_code)
    timestamp_name = protein_order.get(int(protein_prefix))
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
    "_cit1": "Baseline",
    "_cit2": "Target"
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


def hourly_input_to_display(hourly_input_one_case, emeda_input_one_case):
    combined_input_one_case = pd.concat([hourly_input_one_case, emeda_input_one_case])
    features = list(hourly_name.values())
    prefixes = list(hourly_order.values())
    hourly_display_one_case = pd.DataFrame(index=features, columns=prefixes)
    for code in combined_input_one_case.index:
        name, timestamp = get_hourly_display_name(code)
        if name is None or timestamp is None:
            continue
        hourly_display_one_case.loc[name, timestamp] = combined_input_one_case[code]
    return hourly_display_one_case

def hourly_calculate_delta(hourly_display_one_case):
    hourly_calculated_delta = pd.DataFrame(
        index=[hourly_calculated_name['Delta PO2'], hourly_calculated_name['Calc Delta PCO2']],
        columns=hourly_display_one_case.columns,
    )
    delta_po2 = hourly_display_one_case.loc[hourly_name['LA PO2']] - hourly_display_one_case.loc[hourly_name['PA PO2']]
    delta_pco2 = hourly_display_one_case.loc[hourly_name['LA PCO2']] - hourly_display_one_case.loc[hourly_name['PA PCO2']]
    hourly_calculated_delta.loc[hourly_calculated_name['Delta PO2']] = delta_po2
    hourly_calculated_delta.loc[hourly_calculated_name['Calc Delta PCO2']] = delta_pco2
    return hourly_calculated_delta

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
        if name is None or timestamp is None:
            continue
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
        if feature_name is None or timestamp is None:
            continue
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