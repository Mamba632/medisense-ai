import streamlit as st
import plotly.express as px

from app.components.model_loader import load_data
from app.components.charts import (
    correlation_heatmap,
    distribution_plot,
    missing_values_chart,
    class_distribution_chart,
)
from app.components.styles import metric_card
from src.config import DATASETS


def render(dataset_name):
    st.header("EDA Explorer")

    df = load_data(dataset_name)
    target = DATASETS[dataset_name]["target"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Distributions",
            "Correlations",
            "Scatter Matrix",
            "Missing Values",
        ]
    )

    with tab1:
        # Metric cards
        total_missing = int(df.isnull().sum().sum())
        class_balance = df[target].value_counts(normalize=True).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                metric_card("Total Rows", str(df.shape[0])), unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                metric_card("Features", str(df.shape[1] - 1), "success"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card(
                    "Missing Values",
                    str(total_missing),
                    "danger" if total_missing > 0 else "success",
                ),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                metric_card("Majority Class", f"{class_balance:.1%}"),
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with c2:
            st.subheader("Class Distribution")
            fig = class_distribution_chart(df, target)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        selected_cols = st.multiselect(
            "Select features to plot",
            numeric_cols,
            default=numeric_cols[:3],
        )
        color_by_class = st.checkbox("Color by disease class", value=True)

        for col in selected_cols:
            hue = target if color_by_class else None
            fig = distribution_plot(df, col, hue=hue)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap")
        fig = correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

        # Feature-target correlations
        st.subheader("Feature-Target Correlation")
        numeric_df = df.select_dtypes(include="number")
        if target in numeric_df.columns:
            corr_with_target = numeric_df.corr()[target].drop(target).sort_values()
            fig = px.bar(
                x=corr_with_target.values,
                y=corr_with_target.index,
                orientation="h",
                title=f"Correlation with {target}",
                labels={"x": "Correlation", "y": "Feature"},
                color=corr_with_target.values,
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Scatter Matrix")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        n_features = st.slider("Number of features", 3, min(8, len(numeric_cols)), 5)
        selected = numeric_cols[:n_features]
        fig = px.scatter_matrix(
            df,
            dimensions=selected,
            color=target,
            opacity=0.5,
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Missing Values Analysis")
        fig = missing_values_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in this dataset!")

        # Missing pattern heatmap
        missing_total = df.isnull().sum()
        if missing_total.sum() > 0:
            st.subheader("Missing Value Pattern")
            missing_df = df.isnull().astype(int)
            fig = px.imshow(
                missing_df.T,
                color_continuous_scale=["white", "#E63946"],
                title="Missing Value Heatmap (red = missing)",
                aspect="auto",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
