import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

COLLECTION_DIR = ROOT_DIR / "outputs" / "datasets" / "collection"
CLEANED_DIR = ROOT_DIR / "outputs" / "datasets" / "cleaned"
FEATURED_DIR = ROOT_DIR / "outputs" / "datasets" / "featured"
PIPELINE_DIR = ROOT_DIR / "outputs" / "ml_pipeline" / "predict_sale_price" / "v1"


@st.cache_data
def load_house_prices_records():
    """
    Load original collected housing records.
    """
    return pd.read_csv(COLLECTION_DIR / "house_prices_records.csv")


@st.cache_data
def load_saleprice_correlation_summary():
    """
    Load SalePrice correlation summary from the EDA notebook.
    """
    return pd.read_csv(COLLECTION_DIR / "saleprice_correlation_summary.csv", index_col=0)


@st.cache_data
def load_cleaned_inherited_houses():
    """
    Load cleaned inherited houses with readable categorical values.
    """
    return pd.read_csv(
        CLEANED_DIR / "InheritedHousesCleaned.csv",
        keep_default_na=False
    )


@st.cache_data
def load_cleaned_train_set():
    """
    Load cleaned train set with readable categorical values.
    """
    return pd.read_csv(
        CLEANED_DIR / "TrainSetCleaned.csv",
        keep_default_na=False
    )


@st.cache_data
def load_inherited_houses_predictions():
    """
    Load inherited houses with predicted sale prices.
    """
    return pd.read_csv(
        PIPELINE_DIR / "inherited_houses_predictions.csv",
        keep_default_na=False
    )


@st.cache_data
def load_feature_importance():
    """
    Load feature importance table.
    """
    return pd.read_csv(PIPELINE_DIR / "feature_importance.csv")


@st.cache_data
def load_model_comparison():
    """
    Load model comparison dataframe.
    """
    return joblib.load(PIPELINE_DIR / "model_comparison.pkl")


@st.cache_data
def load_model_performance():
    """
    Load final model performance dataframe.
    """
    return joblib.load(PIPELINE_DIR / "model_performance.pkl")


@st.cache_resource
def load_regression_pipeline():
    """
    Load fitted regression pipeline.
    """
    return joblib.load(PIPELINE_DIR / "regression_pipeline.pkl")


@st.cache_resource
def load_feature_engineering_artifacts():
    """
    Load feature-engineering artifacts.
    """
    selected_features = joblib.load(PIPELINE_DIR / "selected_features.pkl")
    mappings = joblib.load(PIPELINE_DIR / "feature_engineering_mappings.pkl")
    config = joblib.load(PIPELINE_DIR / "feature_engineering_config.pkl")

    return selected_features, mappings, config