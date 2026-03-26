import streamlit as st
import pandas as pd
import plotly.express as px

from app.components.model_loader import load_data
from app.components.charts import class_distribution_chart
from app.components.styles import metric_card, section_header
from src.config import DATASETS


def render(dataset_name):
    st.header("Patient Insights")

    df = load_data(dataset_name)
    target = DATASETS[dataset_name]["target"]
    # Population overview
    st.markdown(section_header("Population Overview"), unsafe_allow_html=True)

    disease_rate = df[target].mean()
    age_col = "Age" if "Age" in df.columns else "age"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card("Total Patients", str(len(df))),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card("Disease Rate", f"{disease_rate:.1%}", "danger"),
            unsafe_allow_html=True,
        )
    with c3:
        if age_col in df.columns:
            st.markdown(
                metric_card(
                    "Age Range", f"{int(df[age_col].min())} - {int(df[age_col].max())}"
                ),
                unsafe_allow_html=True,
            )
    with c4:
        if "Gender" in df.columns:
            male_pct = (df["Gender"] == "Male").mean()
            st.markdown(
                metric_card("Male %", f"{male_pct:.0%}", "success"),
                unsafe_allow_html=True,
            )
        elif "sex" in df.columns:
            male_pct = df["sex"].mean()
            st.markdown(
                metric_card("Male %", f"{male_pct:.0%}", "success"),
                unsafe_allow_html=True,
            )

    # Demographic breakdown
    st.markdown(section_header("Demographic Breakdown"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if age_col in df.columns:
            fig = px.histogram(
                df,
                x=age_col,
                color=target,
                barmode="overlay",
                opacity=0.7,
                nbins=20,
                title="Age Distribution by Disease Status",
                color_discrete_sequence=["#00B4D8", "#E63946"],
            )
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = class_distribution_chart(df, target, "Disease Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Age group analysis
    if age_col in df.columns:
        st.markdown(section_header("Age Group Analysis"), unsafe_allow_html=True)

        df_copy = df.copy()
        bins = [0, 30, 45, 60, 120]
        labels = ["<30", "30-45", "45-60", "60+"]
        df_copy["Age Group"] = pd.cut(df_copy[age_col], bins=bins, labels=labels)

        age_disease = (
            df_copy.groupby("Age Group", observed=True)[target].mean().reset_index()
        )
        age_disease.columns = ["Age Group", "Disease Rate"]

        fig = px.bar(
            age_disease,
            x="Age Group",
            y="Disease Rate",
            title="Disease Rate by Age Group",
            color="Disease Rate",
            color_continuous_scale=["#00B4D8", "#E63946"],
        )
        fig.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature statistics by class
    st.markdown(section_header("Feature Statistics by Class"), unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include="number")
    grouped = numeric_df.groupby(df[target]).mean()
    st.dataframe(grouped.style.format("{:.2f}"), use_container_width=True)

    # Box plots
    st.markdown(section_header("Feature Analysis"), unsafe_allow_html=True)
    numeric_cols = [c for c in numeric_df.columns if c != target]

    selected = st.selectbox("Select feature for detailed view", numeric_cols)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(
            df,
            x=target,
            y=selected,
            color=target,
            title=f"{selected} — Box Plot by Class",
            color_discrete_sequence=["#00B4D8", "#E63946"],
        )
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(
            df,
            x=target,
            y=selected,
            color=target,
            box=True,
            title=f"{selected} — Violin Plot by Class",
            color_discrete_sequence=["#00B4D8", "#E63946"],
        )
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
