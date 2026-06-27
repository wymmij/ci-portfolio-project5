import numpy as np
import pandas as pd
import streamlit as st

from src.data_management import (
    load_cleaned_train_set,
    load_feature_engineering_artifacts,
    load_regression_pipeline,
)


FEATURE_LABELS = {
    "OverallQual": "Overall material and finish quality",
    "OverallCond": "Overall house condition",
    "KitchenQual": "Kitchen quality",
    "BsmtExposure": "Basement exposure",
    "BsmtFinType1": "Basement finished area rating",
    "GrLivArea": "Above-ground living area",
    "1stFlrSF": "First floor area",
    "2ndFlrSF": "Second floor area",
    "TotalBsmtSF": "Total basement area",
    "BsmtFinSF1": "Finished basement area",
    "BsmtUnfSF": "Unfinished basement area",
    "BedroomAbvGr": "Bedrooms above grade",
    "LotArea": "Lot area",
    "LotFrontage": "Street frontage",
    "MasVnrArea": "Masonry veneer area",
    "GarageFinish": "Garage finish",
    "GarageArea": "Garage area",
    "GarageYrBlt": "Year garage was built",
    "OpenPorchSF": "Open porch area",
    "EnclosedPorch": "Enclosed porch area",
    "WoodDeckSF": "Wood deck area",
    "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodelled or added",
}


FEATURE_HELP = {
    "OverallQual": "Rates the overall material and finish of the house. 1 = Very Poor, 10 = Very Excellent.",
    "OverallCond": (
        "Rates the overall condition of the house. "
        "The metadata scale is 1 = Very Poor to 10 = Very Excellent, "
        "but the training data ranges from 1 to 9, so the dashboard input is limited to 9."
    ),
    "KitchenQual": "Kitchen quality. The model uses the original Ames dataset category codes.",
    "BsmtExposure": "Refers to walkout or garden-level basement walls.",
    "BsmtFinType1": "Rating of the basement finished area.",
    "GrLivArea": "Above-grade living area in square feet.",
    "1stFlrSF": "First floor area in square feet.",
    "2ndFlrSF": "Second floor area in square feet. Use 0 if there is no second floor.",
    "TotalBsmtSF": "Total basement area in square feet. Use 0 if there is no basement.",
    "BsmtFinSF1": "Type 1 finished basement area in square feet.",
    "BsmtUnfSF": "Unfinished basement area in square feet.",
    "BedroomAbvGr": "Bedrooms above grade. This does not include basement bedrooms.",
    "LotArea": "Lot size in square feet.",
    "LotFrontage": "Linear feet of street connected to the property.",
    "MasVnrArea": "Masonry veneer area in square feet.",
    "GarageFinish": "Interior finish of the garage.",
    "GarageArea": "Garage size in square feet. Use 0 if there is no garage.",
    "GarageYrBlt": "Year the garage was built. Use 0 if there is no garage.",
    "OpenPorchSF": "Open porch area in square feet.",
    "EnclosedPorch": "Enclosed porch area in square feet.",
    "WoodDeckSF": "Wood deck area in square feet.",
    "YearBuilt": "Original construction date.",
    "YearRemodAdd": "Remodel date. Same as construction date if there has been no remodelling or addition.",
}


CATEGORY_LABELS = {
    "KitchenQual": {
        "Ex": "Excellent",
        "Gd": "Good",
        "TA": "Typical/Average",
        "Fa": "Fair",
        "Po": "Poor",
    },
    "BsmtExposure": {
        "Gd": "Good exposure",
        "Av": "Average exposure",
        "Mn": "Minimum exposure",
        "No": "No exposure",
        "None": "No basement",
    },
    "BsmtFinType1": {
        "GLQ": "Good living quarters",
        "ALQ": "Average living quarters",
        "BLQ": "Below average living quarters",
        "Rec": "Average rec room",
        "LwQ": "Low quality",
        "Unf": "Unfinished",
        "None": "No basement",
    },
    "GarageFinish": {
        "Fin": "Finished",
        "RFn": "Rough finished",
        "Unf": "Unfinished",
        "None": "No garage",
    },
}


CATEGORY_ORDERS = {
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
}


def get_feature_label(col):
    """
    Return a human-readable label for a feature.
    """
    return FEATURE_LABELS.get(col, col)


def get_feature_help(col):
    """
    Return help text for a feature widget.
    """
    return FEATURE_HELP.get(col, None)


def format_category_option(col, value):
    """
    Format categorical option labels while preserving the original value.
    """
    category_labels = CATEGORY_LABELS.get(col, {})
    value_label = category_labels.get(value, value)

    return f"{value} — {value_label}"


def get_ordered_category_options(col, values):
    """
    Return category options in a meaningful ordinal order where available.
    """
    values = list(values)

    if col not in CATEGORY_ORDERS:
        return sorted(values)

    ordered_values = [
        value for value in CATEGORY_ORDERS[col]
        if value in values
    ]

    remaining_values = sorted([
        value for value in values
        if value not in ordered_values
    ])

    return ordered_values + remaining_values


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

    user_input = {}

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