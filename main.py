"""
main.py - End-to-end pipeline: load → engineer → train → evaluate → save.

Run this from the project root:
    python main.py

This is the "glue" script. Each step is handled by its own module
so you can also run/test parts independently.
"""

import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_raw_data
from src.feature_engineering import add_features
from src.model_training import prepare_data, train_and_compare
from src.evaluation import evaluate_model, plot_feature_importance
from src.preprocessing import get_feature_names
from src.utils import save_model, get_logger

logger = get_logger("main")

DATA_PATH   = "data/raw.csv"
MODEL_PATH  = "models/churn_model.pkl"
OUTPUT_DIR  = "models"          # also used for saving evaluation plots


def run():
    logger.info("=" * 55)
    logger.info("  Customer Churn Prediction — Training Pipeline")
    logger.info("=" * 55)

    # 1. Load data
    df = load_raw_data(DATA_PATH)

    # 2. Feature engineering + train/test split
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 3. Train models and pick the best one
    best_pipeline, best_name, all_results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    # 4. Full evaluation of the best model
    evaluate_model(
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        model_name=best_name,
        save_dir=OUTPUT_DIR,
    )

    # 5. Feature importance (only prints for tree-based models)
    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        preprocessor.fit(X_train)  # already fit, but get_feature_names needs it
        feature_names = get_feature_names(preprocessor)
    except Exception:
        feature_names = []

    plot_feature_importance(best_pipeline, feature_names, top_n=15, save_dir=OUTPUT_DIR)

    # 6. Save model
    save_model(best_pipeline, MODEL_PATH)

    logger.info("\nAll done! Model saved to models/churn_model.pkl")
    logger.info("Run the Streamlit app with: streamlit run app/app.py")


if __name__ == "__main__":
    run()
