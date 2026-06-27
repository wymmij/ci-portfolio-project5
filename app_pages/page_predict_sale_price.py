import streamlit as st

from src.data_management import load_inherited_houses_predictions
from src.machine_learning.predictive_analysis_ui import (
    display_prediction_input_widgets,
    predict_sale_price,
)


def page_predict_sale_price_body():
    """
    Predict sale price page.
    """
    st.title("Predict Sale Price")

    st.header("Inherited Houses Predictions")

    inherited_predictions_df = load_inherited_houses_predictions()

    st.write(
        "The table below shows predicted sale prices for the four inherited "
        "houses using the final fitted regression model."
    )

    st.dataframe(inherited_predictions_df)

    total_value = inherited_predictions_df["PredictedSalePrice"].sum()

    st.metric(
        label="Total predicted value of inherited houses",
        value=f"${total_value:,.2f}"
    )

    st.header("Predict Sale Price for a Custom House")

    st.write(
        "Use the form below to enter house attributes and generate a sale "
        "price prediction."
    )

    with st.expander("How to use the prediction form", expanded=True):
        st.markdown(
            """
            * Categorical fields use the original Ames housing dataset codes. Each
            dropdown shows the code followed by its meaning, for example
            `TA — Typical/Average`.

            * Numerical fields are pre-filled with median values from the cleaned
            training data. Adjust these values as needed before submitting the form.

            * Input ranges are limited to the values observed in the cleaned training
            data. This avoids asking the model to predict from values outside the
            range it learned from.

            * The prediction applies the same feature engineering steps and fitted
            model created in the project notebooks.
            """
        )

    st.markdown(
        """
        <style>
        div[data-testid="stForm"] {
            max-width: 750px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.form("sale_price_prediction_form"):
        input_df = display_prediction_input_widgets()

        submitted = st.form_submit_button(
            "Predict Sale Price",
            use_container_width=True
        )

    if submitted:
        prediction = predict_sale_price(input_df)[0]

        st.success(f"Predicted sale price: ${prediction:,.2f}")

        with st.expander("Input data used for prediction"):
            st.dataframe(input_df)
