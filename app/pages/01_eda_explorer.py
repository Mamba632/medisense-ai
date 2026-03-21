import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_raw_dataset
from src.config import DATASETS
from app.components.charts import correlation_heatmap, distribution_plot
from app.components.sidebar import dataset_selector, show_dataset_info

st.header("EDA Explorer")

dataset_name = dataset_selector()
show_dataset_info(dataset_name)

df = load_raw_dataset(dataset_name)
target = DATASETS[dataset_name]["target"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Distributions", "Correlations", "Scatter Matrix"]
)

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric(
        "Class Balance",
        f"{df[target].value_counts(normalize=True).iloc[0]:.1%}",
    )
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_col = st.selectbox("Select feature", numeric_cols)
    fig = distribution_plot(df, selected_col, hue=target)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation Heatmap")
    fig = correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Scatter Matrix")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:6]
    import plotly.express as px

    fig = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color=target,
        opacity=0.5,
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)
