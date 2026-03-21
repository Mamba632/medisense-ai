import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def correlation_heatmap(df, title="Feature Correlation Heatmap"):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title=title,
        aspect="auto",
    )
    fig.update_layout(height=600)
    return fig


def distribution_plot(df, column, hue=None, title=None):
    title = title or f"Distribution of {column}"
    fig = px.histogram(
        df,
        x=column,
        color=hue,
        marginal="box",
        title=title,
        barmode="overlay",
        opacity=0.7,
    )
    return fig


def roc_curve_plot(fpr_dict, tpr_dict, auc_dict, title="ROC Curves"):
    fig = go.Figure()
    for name in fpr_dict:
        fig.add_trace(
            go.Scatter(
                x=fpr_dict[name],
                y=tpr_dict[name],
                name=f"{name} (AUC={auc_dict[name]:.3f})",
                mode="lines",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            mode="lines",
            line={"dash": "dash", "color": "gray"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )
    return fig


def risk_gauge(probability, title="Health Risk Score"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=350)
    return fig


def feature_importance_bar(feature_names, importances, title="Feature Importance"):
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(15)
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
    )
    fig.update_layout(height=500)
    return fig
