"""
evaluation.py - Model evaluation and visualization utilities.

All the charts and metrics we need to understand how the model
is actually performing in business terms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts/servers

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
import os
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(pipeline, X_test, y_test, model_name: str = "Model", save_dir: str = None):
    """
    Runs full evaluation: prints metrics + saves confusion matrix and ROC curve.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*55)
    print(f"  Evaluation Report: {model_name}")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"  ROC-AUC Score: {auc:.4f}")
    print("="*55)

    # -- Business interpretation
    report = classification_report(y_test, y_pred, output_dict=True)
    recall = report["1"]["recall"]
    precision = report["1"]["precision"]
    
    print("\n📊 Business Interpretation:")
    print(f"  → Recall {recall:.0%}: Of all customers who actually churned, we correctly")
    print(f"    identified {recall:.0%} of them. Higher = fewer missed churners.")
    print(f"  → Precision {precision:.0%}: When we predict churn, we're right {precision:.0%} of the time.")
    print(f"    Lower precision = more false alarms (intervention cost).")
    print(f"  → AUC {auc:.4f}: Overall ranking ability. 0.5 = random, 1.0 = perfect.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_confusion_matrix(y_test, y_pred, model_name, save_dir)
        _plot_roc_curve(y_test, y_proba, auc, model_name, save_dir)

    return {"auc": auc, "recall": recall, "precision": precision}


def _plot_confusion_matrix(y_test, y_pred, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix → {path}")


def _plot_roc_curve(y_test, y_proba, auc, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#bdc3c7", lw=1.5, linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve → {path}")


def plot_feature_importance(pipeline, feature_names: list, top_n: int = 20, save_dir: str = None):
    """
    Plot feature importance from tree-based models.
    Only works with Random Forest / XGBoost pipelines.
    """
    try:
        clf = pipeline.named_steps["classifier"]
        importances = clf.feature_importances_
    except AttributeError:
        logger.warning("Feature importance not available for this model type.")
        return

    # Pad or trim feature names to match importance array length
    n = len(importances)
    names = feature_names[:n] if len(feature_names) >= n else feature_names + [f"feat_{i}" for i in range(n - len(feature_names))]

    feat_df = pd.DataFrame({"feature": names, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#3498db")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=13)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "feature_importance.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved feature importance → {path}")
    else:
        plt.show()
