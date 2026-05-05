"""
model_training.py - Train multiple models and pick the best one.

Using Pipeline objects so that preprocessing is baked into each model.
This is important for deployment — you don't want to manually re-run
transformations every time you get new data.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

from src.utils import get_logger, save_model
from src.preprocessing import build_preprocessor, NUMERIC_COLS, CATEGORICAL_COLS
from src.feature_engineering import add_features, get_engineered_numeric_cols

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Skipping XGBoost model.")


def get_models(preprocessor) -> dict:
    """
    Define the candidate models.

    Logistic Regression: Good baseline. Interpretable. Fast.
    Random Forest: Handles non-linearities, robust to outliers.
    XGBoost: Usually the best performer on tabular data if tuned.
    
    Not doing hyperparameter tuning here to keep it simple, but
    these defaults are reasonable for this dataset size.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # handles class imbalance (73% No, 27% Yes)
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                scale_pos_weight=3,  # roughly neg/pos ratio to handle imbalance
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0
            ))
        ])

    return models


def train_and_compare(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models, print comparison table, return the best model + name.
    
    Best model is chosen by ROC-AUC score since we care about ranking
    customers by churn risk, not just raw accuracy.
    (Accuracy is misleading with imbalanced classes.)
    """
    preprocessor = build_preprocessor()
    models = get_models(preprocessor)
    
    results = {}

    for name, pipeline in models.items():
        logger.info(f"Training: {name}...")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            "pipeline": pipeline,
            "accuracy": report["accuracy"],
            "precision_churn": report["1"]["precision"],
            "recall_churn": report["1"]["recall"],
            "f1_churn": report["1"]["f1-score"],
            "roc_auc": auc,
        }
        
        logger.info(f"  Accuracy: {report['accuracy']:.4f} | AUC: {auc:.4f} | "
                    f"Recall(Churn): {report['1']['recall']:.4f}")

    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:>9.4f} {r['precision_churn']:>10.4f} "
              f"{r['recall_churn']:>8.4f} {r['f1_churn']:>8.4f} {r['roc_auc']:>8.4f}")
    print("="*70)

    # Pick best by AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipeline = results[best_name]["pipeline"]
    
    print(f"\n[✓] Best model: {best_name} (AUC = {results[best_name]['roc_auc']:.4f})")
    print(f"    Reasoning: AUC measures how well we rank churners vs non-churners,")
    print(f"    which matters more than accuracy with imbalanced classes (73/27 split).")
    
    return best_pipeline, best_name, results


def prepare_data(df: pd.DataFrame):
    """
    Full data prep pipeline: feature engineering → split → return.
    """
    from src.data_loader import get_feature_target_split

    # Add engineered features before splitting
    df = add_features(df)

    X, y = get_feature_target_split(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    logger.info(f"Churn rate in train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test
