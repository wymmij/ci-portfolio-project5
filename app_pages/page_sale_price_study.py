import plotly.express as px
import streamlit as st

from src.data_management import (
    load_house_prices_records,
    load_saleprice_correlation_summary,
)
from src.feature_metadata import create_feature_glossary, format_feature_label, get_feature_label


def page_sale_price_study_body():
    """
    Sale price study page.
    """
    st.title("Sale Price Study")

    df = load_house_prices_records()
    corr_df = load_saleprice_correlation_summary()

    st.write(
        "This page summarises the exploratory data analysis carried out in "
        "the Sale Price Study notebook."
    )

    st.header("Top Correlated Features")

    top_corr = corr_df.sort_values(by="spearman_abs", ascending=False).head(10)

    top_corr_display = top_corr.reset_index().rename(columns={"index": "Feature"})
    top_corr_display.insert(
        1,
        "Meaning",
        top_corr_display["Feature"].apply(get_feature_label)
    )

    st.dataframe(top_corr_display)

    top_corr_plot = top_corr.reset_index().rename(columns={"index": "Feature"})
    top_corr_plot["FeatureLabel"] = top_corr_plot["Feature"].apply(
        lambda feature: f"{get_feature_label(feature)} ({feature})"
    )

    fig = px.bar(
        top_corr_plot,
        x="spearman_abs",
        y="FeatureLabel",
        orientation="h",
        title="Top Features by Absolute Spearman Correlation with SalePrice",
        labels={
            "spearman_abs": "Absolute Spearman correlation",
            "FeatureLabel": "Feature"
        }
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.header("Key Relationships")

    st.subheader("Overall Quality and Sale Price")
    fig = px.box(
        df,
        x="OverallQual",
        y="SalePrice",
        title="SalePrice by Overall Quality"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Above-Ground Living Area and Sale Price")
    fig = px.scatter(
        df,
        x="GrLivArea",
        y="SalePrice",
        title="SalePrice by Above-Ground Living Area"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year Built and Sale Price")
    fig = px.scatter(
        df,
        x="YearBuilt",
        y="SalePrice",
        title="SalePrice by Year Built"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "The strongest relationships found in the study were **overall quality** "
        "(`OverallQual`), **above-ground living area** (`GrLivArea`), **construction year** "
        "(`YearBuilt`), **garage area** (`GarageArea`), and **basement area** (`TotalBsmtSF`)."
    )