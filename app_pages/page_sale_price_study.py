import plotly.express as px
import streamlit as st

from src.data_management import (
    load_house_prices_records,
    load_saleprice_correlation_summary,
)


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

    st.dataframe(top_corr)

    fig = px.bar(
        top_corr.reset_index(),
        x="spearman_abs",
        y="index",
        orientation="h",
        title="Top Features by Absolute Spearman Correlation with SalePrice",
        labels={
            "spearman_abs": "Absolute Spearman correlation",
            "index": "Feature"
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
        "The strongest relationships found in the study were overall quality, "
        "living area, construction year, garage area, and basement area."
    )