import streamlit as st
from src.config import DATASETS


def dataset_selector():
    dataset_name = st.sidebar.selectbox(
        "Select Disease Dataset",
        options=list(DATASETS.keys()),
        format_func=lambda x: DATASETS[x]["description"],
    )
    return dataset_name


def show_dataset_info(dataset_name):
    config = DATASETS[dataset_name]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Target:** `{config['target']}`")
    st.sidebar.markdown(f"**Numeric features:** {len(config['numeric_features'])}")
    st.sidebar.markdown(
        f"**Categorical features:** {len(config['categorical_features'])}"
    )
