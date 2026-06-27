import pandas as pd


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
    "HouseAge": "House age",
    "RemodAge": "Remodelling age",
    "GarageAge": "Garage age",
}


FEATURE_HELP = {
    "OverallQual": (
        "Rates the overall material and finish of the house. "
        "The metadata scale is 1 = Very Poor to 10 = Very Excellent."
    ),
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
    "HouseAge": "Age of the house, calculated from the reference year used in feature engineering.",
    "RemodAge": "Age of the most recent remodelling or addition, calculated from the reference year.",
    "GarageAge": "Age of the garage, calculated from the reference year. A value of 0 is used where there is no garage.",
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


def get_feature_label(feature: str) -> str:
    """
    Return a human-readable feature label.
    """
    return FEATURE_LABELS.get(feature, feature)


def get_feature_help(feature: str) -> str | None:
    """
    Return feature help text where available.
    """
    return FEATURE_HELP.get(feature)


def format_feature_label(feature: str) -> str:
    """
    Return a readable feature label with the original dataset code.
    """
    label = get_feature_label(feature)

    if label == feature:
        return feature

    return f"{label} (`{feature}`)"


def format_category_option(feature: str, value: str) -> str:
    """
    Format categorical option labels while preserving the original value.
    """
    category_labels = CATEGORY_LABELS.get(feature, {})
    value_label = category_labels.get(value, value)

    return f"{value} — {value_label}"


def get_ordered_category_options(feature: str, values) -> list[str]:
    """
    Return category options in a meaningful ordinal order where available.
    """
    values = [str(value) for value in values]

    if feature not in CATEGORY_ORDERS:
        return sorted(values)

    ordered_values = [
        value for value in CATEGORY_ORDERS[feature]
        if value in values
    ]

    remaining_values = sorted([
        value for value in values
        if value not in ordered_values
    ])

    return ordered_values + remaining_values


def create_feature_glossary(features: list[str]) -> pd.DataFrame:
    """
    Create a glossary dataframe for selected features.
    """
    return pd.DataFrame({
        "Feature": features,
        "Meaning": [get_feature_label(feature) for feature in features],
    })