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
    colors = px.colors.qualitative.Set2
    for i, name in enumerate(fpr_dict):
        fig.add_trace(
            go.Scatter(
                x=fpr_dict[name],
                y=tpr_dict[name],
                name=f"{name} (AUC={auc_dict[name]:.3f})",
                mode="lines",
                line={"color": colors[i % len(colors)], "width": 2},
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
        plot_bgcolor="white",
    )
    return fig


def risk_gauge(probability, title="Health Risk Score"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0077B6"},
                "steps": [
                    {"range": [0, 30], "color": "#D4EDDA"},
                    {"range": [30, 70], "color": "#FFF3CD"},
                    {"range": [70, 100], "color": "#F8D7DA"},
                ],
                "threshold": {
                    "line": {"color": "#E63946", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin={"t": 60, "b": 20, "l": 30, "r": 30})
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
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=500, showlegend=False, plot_bgcolor="white")
    return fig


def confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Disease", "Disease"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
        labels={"x": "Predicted", "y": "Actual"},
    )
    fig.update_layout(height=400, plot_bgcolor="white")
    return fig


def shap_bar_chart(shap_values, feature_names, top_n=10):
    df = pd.DataFrame({"feature": feature_names, "shap_value": shap_values})
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=True).tail(top_n)
    colors = ["#E63946" if v > 0 else "#0077B6" for v in df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Feature Contributions to Prediction",
        xaxis_title="SHAP Value (impact on risk)",
        height=400,
        plot_bgcolor="white",
        margin={"l": 120},
    )
    return fig


def missing_values_chart(df, title="Missing Values per Feature"):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) == 0:
        return None
    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        title=title,
        labels={"x": "Missing Count", "y": "Feature"},
        color_discrete_sequence=["#E63946"],
    )
    fig.update_layout(height=max(300, len(missing) * 40), plot_bgcolor="white")
    return fig


def class_distribution_chart(df, target_col, title="Class Distribution"):
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].map({0: "No Disease", 1: "Disease"})
    fig = px.pie(
        counts,
        values="Count",
        names="Class",
        title=title,
        color_discrete_sequence=["#00B4D8", "#E63946"],
        hole=0.4,
    )
    fig.update_layout(height=350)
    return fig


def parallel_coordinates_plot(df, features, color_col):
    fig = px.parallel_coordinates(
        df,
        dimensions=features,
        color=color_col,
        color_continuous_scale=["#00B4D8", "#E63946"],
        title="Parallel Coordinates — Feature Comparison by Class",
    )
    fig.update_layout(height=500)
    return fig


def model_comparison_table(metrics_dict):
    rows = []
    for model_name, metrics in metrics_dict.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1 Score": f"{metrics['f1']:.4f}",
                "ROC-AUC": f"{metrics.get('roc_auc', 'N/A')}",
            }
        )
    return pd.DataFrame(rows)
