"""
feature_engineering.py - Creates new features from existing ones.

I created these features based on business intuition + what I saw in EDA.
They're not random — each one has a clear reason.
"""

import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 3 engineered features to the dataframe.

    Returns a copy (don't mutate the original — annoying bug to track down).
    """
    df = df.copy()

    # --- Feature 1: Charges per month of tenure
    # Customers who pay more relative to how long they've been around
    # might feel the product isn't worth it. New expensive plan = risky.
    # Avoiding division by zero for brand-new customers (tenure=0)
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # --- Feature 2: Is the customer on a high-cost plan?
    # Simple binary flag. From EDA, customers paying > $65/month churn more.
    # The threshold isn't perfectly tuned but it captures the general pattern.
    df["is_high_value"] = (df["MonthlyCharges"] > 65).astype(int)

    # --- Feature 3: Number of additional services subscribed
    # Customers with more services are more "locked in" to the ecosystem
    # and tend to churn less (they'd have to replace everything).
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Count services where the value is "Yes"
    df["num_services"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    logger.info(f"Feature engineering complete. New features: charges_per_tenure, is_high_value, num_services")
    return df


def get_engineered_numeric_cols():
    """Returns column names of the newly created numeric features."""
    return ["charges_per_tenure", "is_high_value", "num_services"]
