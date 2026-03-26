import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

from app.components.model_loader import (
    load_model,
    load_preprocessor,
    load_data,
    is_model_available,
)
from app.components.charts import (
    roc_curve_plot,
    confusion_matrix_plot,
    feature_importance_bar,
    model_comparison_table,
)
from app.components.styles import metric_card
from src.data.loader import get_target_column
from src.data.splitter import stratified_split
from src.evaluation.metrics import compute_metrics


def render(dataset_name):
    st.header("Model Performance")

    if not is_model_available(dataset_name):
        st.warning(f"No trained model found for {dataset_name}. Run training first.")
        return

    ensemble = load_model(dataset_name)
    preprocessor = load_preprocessor(dataset_name)

    df = load_data(dataset_name)
    target_col = get_target_column(dataset_name)
    _, _, test_df = stratified_split(df, target_col)

    X_test = preprocessor.transform(test_df.drop(columns=[target_col]))
    y_test = test_df[target_col].values

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ROC Curves", "Confusion Matrices", "Comparison Table", "Feature Importance"]
    )

    with tab1:
        fpr_dict, tpr_dict, auc_dict = {}, {}, {}

        for name, learner in ensemble.fitted_base_learners_:
            y_prob = learner.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fpr_dict[name] = fpr
            tpr_dict[name] = tpr
            auc_dict[name] = roc_auc_score(y_test, y_prob)

        y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
        fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
        fpr_dict["Stacking Ensemble"] = fpr_ens
        tpr_dict["Stacking Ensemble"] = tpr_ens
        auc_dict["Stacking Ensemble"] = roc_auc_score(y_test, y_prob_ens)

        fig = roc_curve_plot(fpr_dict, tpr_dict, auc_dict)
        st.plotly_chart(fig, use_container_width=True)

        # Top AUC metric
        best_model = max(auc_dict, key=auc_dict.get)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                metric_card("Best Model", best_model, "success"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_card("Best AUC", f"{auc_dict[best_model]:.4f}", "warning"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card("Ensemble AUC", f"{auc_dict['Stacking Ensemble']:.4f}"),
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        all_models = list(ensemble.fitted_base_learners_) + [
            ("Stacking Ensemble", ensemble)
        ]
        for i, (name, learner) in enumerate(all_models):
            with cols[i % 3]:
                y_pred = learner.predict(X_test)
                fig = confusion_matrix_plot(y_test, y_pred, title=name)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Model Comparison")
        metrics_dict = {}
        for name, learner in ensemble.fitted_base_learners_:
            y_pred = learner.predict(X_test)
            y_prob = learner.predict_proba(X_test)[:, 1]
            metrics_dict[name] = compute_metrics(y_test, y_pred, y_prob)

        y_pred_ens = ensemble.predict(X_test)
        y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
        metrics_dict["Stacking Ensemble"] = compute_metrics(
            y_test, y_pred_ens, y_prob_ens
        )

        comparison_df = model_comparison_table(metrics_dict)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Feature Importance")
        tree_models = {
            name: learner
            for name, learner in ensemble.fitted_base_learners_
            if hasattr(learner, "feature_importances_")
        }

        if tree_models:
            selected_model = st.radio(
                "Select model",
                list(tree_models.keys()),
                horizontal=True,
            )
            learner = tree_models[selected_model]
            feature_names = preprocessor.get_feature_names_out()
            importances = learner.feature_importances_
            fig = feature_importance_bar(
                feature_names,
                importances,
                title=f"{selected_model} — Feature Importance",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is only available for tree-based models.")
