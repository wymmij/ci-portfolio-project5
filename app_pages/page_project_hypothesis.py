import streamlit as st

from src.data_management import load_saleprice_correlation_summary
from src.feature_metadata import get_feature_label


def page_project_hypothesis_body():
    """
    Project hypothesis page.
    """
    st.title("Project Hypothesis")

    st.header("Hypothesis")

    st.write(
        "Houses with higher overall quality (`OverallQual`), larger above-ground "
        "living area (`GrLivArea`) and newer construction dates (`YearBuilt`) tend "
        "to have higher sale prices."
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

    validation_df = corr_df.loc[available_features][["pearson", "spearman"]].copy()
    validation_df = validation_df.reset_index().rename(columns={"index": "Feature"})
    validation_df.insert(
        1,
        "Meaning",
        validation_df["Feature"].apply(get_feature_label)
    )

    st.dataframe(validation_df)

    st.success(
        "The Sale Price Study supports the hypothesis. Overall quality "
        "(`OverallQual`), above-ground living area (`GrLivArea`), and construction "
        "year (`YearBuilt`) were among the strongest positive correlations with "
        "`SalePrice`."
    )

    st.write(
        "The hypothesis is therefore considered supported by the exploratory "
        "analysis. The result does not prove causation, but it does show that "
        "these attributes are meaningfully associated with sale price in this "
        "dataset."
    )