import streamlit as st


def page_summary_body():
    """
    Project summary page.
    """
    st.title("Heritage Housing Issues")

    st.write(
        "This dashboard supports a study of house sale prices in Ames, Iowa. "
        "The project investigates which house attributes are most associated "
        "with sale price and uses a machine learning model to predict sale "
        "prices for inherited houses."
    )

    st.header("Project Dataset")

    st.write(
        "The dataset contains historical house sale records and a separate "
        "set of four inherited houses that require sale price predictions."
    )

    st.header("Business Requirements")

    st.subheader("Business Requirement 1")
    st.write(
        "The client wants to understand which house attributes are most "
        "associated with sale price."
    )

    st.subheader("Business Requirement 2")
    st.write(
        "The client wants to predict the sale price of four inherited houses "
        "and also estimate the sale price of other houses with similar data."
    )

    st.header("Dashboard Pages")

    st.write(
        "* **Sale Price Study**: explores variables most associated with sale price.\n"
        "* **Project Hypothesis**: states and evaluates the project hypothesis.\n"
        "* **Predict Sale Price**: predicts sale prices for inherited houses and custom inputs.\n"
        "* **ML Performance**: shows model performance, feature importance, and evaluation plots."
    )

    st.info(
        "Use the navigation menu in the sidebar to move through the dashboard."
    )