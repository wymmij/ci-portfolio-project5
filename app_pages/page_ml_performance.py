import streamlit as st
from pathlib import Path

from src.data_management import (
    PIPELINE_DIR,
    load_feature_importance,
    load_model_comparison,
    load_model_performance,
)


def page_ml_performance_body():
    """
    ML performance page.
    """
    st.title("ML Performance")

    st.write(
        "This page summarises the performance of the final regression model "
        "used to predict house sale prices."
    )

    st.header("Model Comparison")

    model_comparison_df = load_model_comparison()
    st.dataframe(model_comparison_df.round(3))

    st.header("Final Model Performance")

    model_performance = load_model_performance()
    st.dataframe(model_performance)

    test_r2 = model_performance.loc[
        model_performance["Set"] == "Test",
        "R2"
    ].iloc[0]

    st.metric("Final test R²", f"{test_r2:.3f}")

    if test_r2 >= 0.75:
        st.success(
            "The model meets the project success threshold of R² >= 0.75."
        )
    else:
        st.error(
            "The model does not meet the project success threshold of R² >= 0.75."
        )

    st.header("Actual vs Predicted Sale Price")

    train_plot = PIPELINE_DIR / "actual_vs_predicted_train.png"
    test_plot = PIPELINE_DIR / "actual_vs_predicted_test.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Set")
        if train_plot.exists():
            st.image(str(train_plot))
        else:
            st.warning("Train actual-vs-predicted plot not found.")

    with col2:
        st.subheader("Test Set")
        if test_plot.exists():
            st.image(str(test_plot))
        else:
            st.warning("Test actual-vs-predicted plot not found.")

    st.header("Feature Importance")

    feature_importance_df = load_feature_importance()

    st.dataframe(feature_importance_df.head(15))

    feature_importance_plot = PIPELINE_DIR / "feature_importance.png"

    if feature_importance_plot.exists():
        st.image(str(feature_importance_plot))
    else:
        st.warning("Feature importance plot not found.")