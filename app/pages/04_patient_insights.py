import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_raw_dataset
from src.config import DATASETS
from app.components.sidebar import dataset_selector, show_dataset_info

st.header("Patient Insights")

dataset_name = dataset_selector()
show_dataset_info(dataset_name)

df = load_raw_dataset(dataset_name)
target = DATASETS[dataset_name]["target"]

# Age distribution by disease status
age_col = "Age" if "Age" in df.columns else "age"
if age_col in df.columns:
    st.subheader("Age Distribution by Disease Status")
    fig = px.histogram(
        df,
        x=age_col,
        color=target,
        barmode="overlay",
        opacity=0.7,
        nbins=20,
        title=f"Age Distribution — {DATASETS[dataset_name]['description']}",
    )
    st.plotly_chart(fig, use_container_width=True)

# Class distribution
st.subheader("Disease Class Distribution")
class_counts = df[target].value_counts().reset_index()
class_counts.columns = [target, "count"]
fig = px.pie(class_counts, values="count", names=target, title="Class Distribution")
st.plotly_chart(fig, use_container_width=True)

# Numeric feature summary
st.subheader("Feature Statistics by Class")
numeric_df = df.select_dtypes(include="number")
grouped = numeric_df.groupby(df[target]).mean()
st.dataframe(grouped.style.format("{:.2f}"), use_container_width=True)

# Box plots
st.subheader("Feature Box Plots")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)
selected = st.selectbox("Select feature for box plot", numeric_cols)
fig = px.box(df, x=target, y=selected, color=target, title=f"{selected} by Class")
st.plotly_chart(fig, use_container_width=True)
