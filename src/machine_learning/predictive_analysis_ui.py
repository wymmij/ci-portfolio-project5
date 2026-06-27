import numpy as np
import pandas as pd
import streamlit as st

from typing import Any

from src.data_management import (
    load_cleaned_train_set,
    load_feature_engineering_artifacts,
    load_regression_pipeline,
)
from src.feature_metadata import (
    format_category_option,
    get_feature_help,
    get_feature_label,
    get_ordered_category_options,
)


def add_age_features(df, reference_year):
    """
    Add age-related features using the stored feature-engineering reference year.
    """
    df = df.copy()

    df["HouseAge"] = reference_year - df["YearBuilt"]
    df["RemodAge"] = reference_year - df["YearRemodAdd"]

    df["GarageAge"] = np.where(
        df["GarageYrBlt"] == 0,
        0,
        reference_year - df["GarageYrBlt"]
    )

    return df


def apply_ordinal_encoding(df, mappings):
    """
    Apply saved ordinal mappings to categorical features.
    """
    df = df.copy()

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def prepare_prediction_data(df):
    """
    Apply the same feature engineering used in the notebooks.
    """
    selected_features, mappings, config = load_feature_engineering_artifacts()

    df = df.copy()

    df = add_age_features(
        df,
        reference_year=config["reference_year"]
    )

    df = apply_ordinal_encoding(df, mappings)

    df = df.drop(columns=config["drop_features"])

    df = df[selected_features]

    return df


def predict_sale_price(input_df):
    """
    Predict sale price for raw user input.
    """
    model = load_regression_pipeline()
    prediction_data = prepare_prediction_data(input_df)

    predictions = model.predict(prediction_data)

    return predictions


def display_prediction_input_widgets():
    """
    Display Streamlit widgets for entering house feature values.
    """
    train_df = load_cleaned_train_set()

    target = "SalePrice"
    feature_df = train_df.drop(columns=[target])

    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.to_list()
    numerical_cols = feature_df.select_dtypes(include=["number"]).columns.to_list()

    user_input: dict[str, Any] = {}

    with st.expander("Quality and condition", expanded=True):
        quality_cols = [
            "OverallQual",
            "OverallCond",
            "KitchenQual",
            "BsmtExposure",
            "BsmtFinType1",
            "GarageFinish"
        ]

        for col in quality_cols:
            if col in categorical_cols:
                options = get_ordered_category_options(
                    col,
                    feature_df[col].dropna().unique()
                )
                user_input[col] = st.selectbox(
                    label=get_feature_label(col),
                    options=options,
                    format_func=lambda value, col=col: format_category_option(col, value),
                    help=get_feature_help(col)
                )

            if col in numerical_cols:
                user_input[col] = st.number_input(
                    label=get_feature_label(col),
                    min_value=int(feature_df[col].min()),
                    max_value=int(feature_df[col].max()),
                    value=int(feature_df[col].median()),
                    step=1,
                    help=get_feature_help(col)
                )

    with st.expander("Area and room features", expanded=True):
        area_cols = [
            "GrLivArea",
            "1stFlrSF",
            "2ndFlrSF",
            "TotalBsmtSF",
            "BsmtFinSF1",
            "BsmtUnfSF",
            "BedroomAbvGr",
            "LotArea",
            "LotFrontage",
            "MasVnrArea"
        ]

        for col in area_cols:
            if col in numerical_cols:
                user_input[col] = st.number_input(
                    label=get_feature_label(col),
                    min_value=float(feature_df[col].min()),
                    max_value=float(feature_df[col].max()),
                    value=float(feature_df[col].median()),
                    step=1.0,
                    help=get_feature_help(col)
                )

    with st.expander("Garage and porch features", expanded=False):
        garage_porch_cols = [
            "GarageArea",
            "GarageYrBlt",
            "OpenPorchSF",
            "EnclosedPorch",
            "WoodDeckSF"
        ]

        for col in garage_porch_cols:
            if col in numerical_cols:
                user_input[col] = st.number_input(
                    label=get_feature_label(col),
                    min_value=float(feature_df[col].min()),
                    max_value=float(feature_df[col].max()),
                    value=float(feature_df[col].median()),
                    step=1.0,
                    help=get_feature_help(col)
                )

    with st.expander("Year features", expanded=False):
        year_cols = [
            "YearBuilt",
            "YearRemodAdd"
        ]

        for col in year_cols:
            if col in numerical_cols:
                user_input[col] = st.number_input(
                    label=get_feature_label(col),
                    min_value=int(feature_df[col].min()),
                    max_value=int(feature_df[col].max()),
                    value=int(feature_df[col].median()),
                    step=1,
                    help=get_feature_help(col)
                )

    already_collected = set(user_input.keys())

    for col in feature_df.columns:
        if col in already_collected:
            continue

        if col in categorical_cols:
            options = get_ordered_category_options(
                col,
                feature_df[col].dropna().unique()
            )
            user_input[col] = st.selectbox(
                label=get_feature_label(col),
                options=options,
                format_func=lambda value, col=col: format_category_option(col, value),
                help=get_feature_help(col)
            )

        elif col in numerical_cols:
            user_input[col] = st.number_input(
                label=get_feature_label(col),
                min_value=float(feature_df[col].min()),
                max_value=float(feature_df[col].max()),
                value=float(feature_df[col].median()),
                step=1.0,
                help=get_feature_help(col)
            )

    input_df = pd.DataFrame([user_input])

    return input_df