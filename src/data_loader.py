"""
data_loader.py - Reads raw CSV and does minimal cleaning before anything else.

Keeping this separate from preprocessing because loading and transforming
are two different responsibilities. Learned that the hard way on a past project.
"""

import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw Telco Churn CSV.

    Notes:
    - TotalCharges comes in as object dtype (has some spaces instead of NaN)
    - We fix that here since it's a loading artifact, not a preprocessing decision
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    logger.info(f"Raw data shape: {df.shape}")

    # TotalCharges has whitespace entries for customers with tenure=0
    # (brand new customers who haven't been billed yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # How many got coerced to NaN?
    n_missing = df["TotalCharges"].isnull().sum()
    if n_missing > 0:
        logger.info(f"Found {n_missing} rows with missing TotalCharges (tenure=0 customers) — will handle in preprocessing")

    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Split into features (X) and target (y).
    
    Drops customerID since it's just an identifier with no predictive value.
    """
    drop_cols = ["customerID", "Churn"]
    X = df.drop(columns=drop_cols)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y


if __name__ == "__main__":
    df = load_raw_data("data/raw.csv")
    print(df.head())
    print(df.dtypes)
