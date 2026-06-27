import streamlit as st

from src.data_management import load_saleprice_correlation_summary


def page_project_hypothesis_body():
    """
    Project hypothesis page.
    """
    st.title("Project Hypothesis")

    st.header("Hypothesis")

    st.write(
        "Houses with higher overall quality, larger above-ground living area, "
        "and newer construction dates tend to have higher sale prices."
    )

    st.header("Validation")

    corr_df = load_saleprice_correlation_summary()

    relevant_features = [
        "OverallQual",
        "GrLivArea",
        "YearBuilt",
        "GarageArea",
        "TotalBsmtSF"
    ]

    available_features = [
        feature for feature in relevant_features
        if feature in corr_df.index
    ]

    st.dataframe(
        corr_df.loc[available_features][["pearson", "spearman"]]
    )

    st.write(
        "The Sale Price Study supports the hypothesis. `OverallQual`, "
        "`GrLivArea`, and `YearBuilt` were among the strongest positive "
        "correlations with `SalePrice`."
    )

    st.write(
        "The hypothesis is therefore considered supported by the exploratory "
        "analysis. The result does not prove causation, but it does show that "
        "these attributes are meaningfully associated with sale price in this "
        "dataset."
    )