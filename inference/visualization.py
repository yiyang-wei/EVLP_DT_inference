import pandas as pd
import plotly.express as px
from .reformat import *


def hourly_per_feature_line_plot(feature_prediction: pd.Series):
    observed = pd.DataFrame({
        "Hour": [1, 2, 3],
        "Value": feature_prediction[["Observed 1st Hour", "Observed 2nd Hour", "Observed 3rd Hour"]],
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Hour": [1, 2, 3],
        "Value": feature_prediction[["Observed 1st Hour", "Predicted 2nd Hour", "Static Predicted 3rd Hour"]],
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Hour": [1, 2, 3],
        "Value": feature_prediction[["Observed 1st Hour", "Observed 2nd Hour", "Dynamic Predicted 3rd Hour"]],
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)
    fig = px.line(
        df,
        x="Hour",
        y="Value",
        color="Type",
        markers=True,
        title="Hourly Observations vs Predictions"
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['1st Hour', '2nd Hour', '3rd Hour']),
        yaxis_title=feature_prediction.name,
        legend_title_text="Observation/Prediction Type"
    )
    return fig


def hourly_all_features_line_plot(hourly_prediction: pd.DataFrame):
    features_to_drop = [hourly_name[code] for code in ["pMean", "pPeak", "Cdyn"]]
    hourly_prediction = hourly_prediction.drop(index=features_to_drop, errors='ignore')
    n_features = len(hourly_prediction.index)
    observed = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[["Observed 1st Hour", "Observed 2nd Hour", "Observed 3rd Hour"]].values.flatten(),
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[["Observed 1st Hour", "Predicted 2nd Hour", "Static Predicted 3rd Hour"]].values.flatten(),
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[["Observed 1st Hour", "Observed 2nd Hour", "Dynamic Predicted 3rd Hour"]].values.flatten(),
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)

    color_map = {
        "Observed": "grey",
        "Static Predicted": "#FD8008",
        "Dynamic Predicted": "#49548A"
    }

    dash_map = {
        "Observed": "solid",
        "Static Predicted": "dot",
        "Dynamic Predicted": "dash"
    }

    y_padding = {
        # 'pMean': 2,
        'pPlat': 2,
        # 'pPeak': 2,
        'PAP': 2,
        'LAP': 2,
        'LA PO2': 20,
        'PA PO2': 10,
        'LA PCO2': 5,
        'PA PCO2': 5,
        'Cstat': 10,
        # 'Cdyn': 10,
        'LA Na+': 10,
        'LA Ca++': 0.5,
        'LA CL': 10,
        'LA K+': 1,
        'LA HCO3': 1,
        'LA BE': 5,
        'LA pH': 0.2,
        'LA Lact': 2,
        'LA Glu': 2,
        'STEEN lost': 50
    }
    y_boundary = {
        'LA BE': (None, -10),
        'STEEN lost': (0, None),
    }

    fig = px.line(
        df,
        x="Hour",
        y="Value",
        color="Type",
        color_discrete_map=color_map,
        line_dash="Type",
        line_dash_map=dash_map,
        facet_col="Feature",
        category_orders={"Feature": [name for name in hourly_name.values() if name not in features_to_drop]},
        facet_col_wrap=6,
        markers=True,
        title="Hourly Observations vs Predictions for Each Feature"
    )

    fig.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
    fig.for_each_xaxis(lambda x: x.update(tickmode='array', tickvals=[1, 2, 3], showticklabels=True))

    fig.update_traces(opacity=0.7, line=dict(width=3), marker=dict(size=8))

    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=100),
        height=800,
    )

    for i, annotation in enumerate(fig.layout.annotations):
        feature = annotation.text
        feature_data = df[df["Feature"] == feature]["Value"]
        padding = y_padding.get(hourly_code[feature])
        if padding is None:
            continue
        lower, upper = y_boundary.get(hourly_code[feature], (0, None))

        y_min = feature_data.min() - padding
        if lower is not None:
            y_min = max(y_min, lower)
        y_max = feature_data.max() + padding
        if upper is not None:
            y_max = min(y_max, upper)

        yaxis_name = f'yaxis{i+1}'
        if yaxis_name in fig.layout:
            fig.layout[yaxis_name].update(range=[y_min, y_max])

    return fig
