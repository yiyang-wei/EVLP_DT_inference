import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import os
import pickle
import pathlib


def image_pc_inference_static(model_folder, data_folder, output_folder):
    model_folder = pathlib.Path(model_folder)
    data_folder = pathlib.Path(data_folder)
    output_folder = pathlib.Path(output_folder)

    feat_dir = model_folder / 'XGB_PC' / 'results_pred_hyper'

    save_dir = output_folder / 'ImagePC' / 'Static'
    model_dir = model_folder / 'XGB_PC' / 'models_pred'

    new_hourly_data_path = data_folder / "hourly_data_simulated.csv"
    hourly_pred_h2_path = output_folder / 'Hourly' / 'H1_to_H2' / 'predicted_Y.csv'
    image_pc1h_path = data_folder / "PC1h_data_simulated.csv"
    image_pc3h_path = data_folder / "PC3h_data_simulated.csv"

    results = []

    n_jobs = 10

    scores = ['neg_mean_absolute_error', 'neg_mean_squared_error']  # 'r2'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # prepare data
    tab_data_df = pd.read_csv(new_hourly_data_path, index_col='Simulated Donor Id')
    tab_data_df_pred = pd.read_csv(hourly_pred_h2_path, index_col='Simulated Donor Id')
    columns_80 = [col for col in tab_data_df.columns if col.startswith('80_')]
    tab_data_df.loc[:, columns_80] = tab_data_df_pred.loc[:, columns_80]

    image_pc1h = pd.read_csv(image_pc1h_path, index_col='Simulated Donor Id')
    image_pc3h = pd.read_csv(image_pc3h_path, index_col='Simulated Donor Id')
    image_pc1h.columns = ['1h_' + col_name for col_name in image_pc1h.columns if col_name != 'Simulated Donor Id']
    image_pc3h.columns = ['3h_' + col_name for col_name in image_pc3h.columns if col_name != 'Simulated Donor Id']

    merged_df = pd.merge(tab_data_df, image_pc1h, on='Simulated Donor Id')
    merged_df = pd.merge(merged_df, image_pc3h, on='Simulated Donor Id')

    cytokines = ['IL-6', 'IL-8', 'IL-10', 'IL-1B']

    merged_df = merged_df[
        [col for col in merged_df.columns if col.startswith('70_') or col.startswith('80_') or ('pca' in col)]]
    merged_df = merged_df[[col for col in merged_df.columns if not any([y in col for y in cytokines])]]
    merged_df = merged_df.dropna()

    merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith('90_')]

    ytest = merged_df[[col_name for col_name in merged_df.columns if '3h_' in col_name]]
    Xtest = merged_df[[col_name for col_name in merged_df.columns if '3h_feature_pca' not in col_name]]
    Xtest = Xtest[[col_name for col_name in Xtest.columns if not 'STEEN' in col_name]]

    results = []

    y_columns = [f'3h_feature_pca_{i}' for i in range(10)]
    for col_name in y_columns:  # '3h_feature_pca_0...9'
        pca_num = int(col_name.replace('3h_feature_pca_', ''))
        features_path = os.path.join(feat_dir, f'top_features_pca_{pca_num}.txt')
        with open(features_path, 'r') as f:
            features = f.read().splitlines()

        top_features_index = pd.Index(features)

        X_test = Xtest[top_features_index]

        id_list = []
        y_trues = []
        y_preds = []

        np.random.seed(42)

        y_test = ytest[col_name]

        with open(os.path.join(model_dir, f'{col_name}_model.pkl'), 'rb') as file:
            pipeline = pickle.load(file)

        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mpe = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        params = {'mse': mse, 'mae': mae, 'mpe': mpe}

        results.append(params)

        y_test_pred = pd.DataFrame(
            {'Simulated Donor Id': y_test.index.tolist(), 'y_test': y_test.tolist(), 'y_pred': y_pred})
        y_test_pred.to_csv(os.path.join(save_dir, f'{col_name}_predictions.csv'), index=False)

    results_df = pd.DataFrame(results)


if __name__ == "__main__":
    Model_folder = pathlib.Path("../Model")
    Data_folder = pathlib.Path("../Data")
    Output_folder = pathlib.Path("../Output")

    image_pc_inference_static(Model_folder, Data_folder, Output_folder)