"""
preprocessing.py - Handles missing values, encoding, and scaling.

Design decision: using sklearn Pipeline + ColumnTransformer so that
the same transformations can be applied consistently during training
and inference. No data leakage this way.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils import get_logger

logger = get_logger(__name__)


# -- Column groupings (determined from EDA)
# SeniorCitizen is already 0/1 so we treat it as numeric
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]


def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that handles:

    Numeric columns:
        - Impute missing with median (only TotalCharges has NaN, but
          better to be safe for production)
        - StandardScaler: makes Logistic Regression converge faster
          and prevents large-magnitude features from dominating

    Categorical columns:
        - Impute missing with most frequent value
        - OneHotEncoder: drop='first' to avoid dummy variable trap
          (important for linear models)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ], remainder="drop")

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after transformation.
    Useful for feature importance plots.
    """
    cat_features = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(CATEGORICAL_COLS)
    return NUMERIC_COLS + list(cat_features)
