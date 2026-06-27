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

    input_df = display_prediction_input_widgets()

    if st.button("Predict Sale Price"):
        prediction = predict_sale_price(input_df)[0]

        st.success(f"Predicted sale price: ${prediction:,.2f}")

        with st.expander("Input data used for prediction"):
            st.dataframe(input_df)