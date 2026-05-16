"""
Evaluation & Visualisation Module
  - PR-AUC (primary metric per paper)
  - ROC-AUC
  - Classification report
  - Confusion matrix heatmap
  - Precision-Recall curves (per class)
  - Feature importance plot
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize


# ── Output directory ─────────────────────────────────────────────────────────
PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Binary PR-AUC ────────────────────────────────────────────────────────────
def pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return average_precision_score(y_true, y_score)


# ── Multiclass macro PR-AUC ──────────────────────────────────────────────────
def pr_auc_multiclass(y_true: np.ndarray, y_score: np.ndarray,
                      classes: np.ndarray) -> dict:
    """
    Returns per-class and macro-average PR-AUC.
    y_score shape : (n_samples, n_classes)
    """
    y_bin = label_binarize(y_true, classes=classes)
    results = {}
    aucs = []
    for i, cls in enumerate(classes):
        col = y_bin[:, i] if y_bin.ndim > 1 else y_bin
        sc  = y_score[:, i] if y_score.ndim > 1 else y_score
        ap  = average_precision_score(col, sc)
        results[cls] = round(ap, 4)
        aucs.append(ap)
    results["macro"] = round(float(np.mean(aucs)), 4)
    return results


# ── Comprehensive metrics summary ────────────────────────────────────────────
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             binary: bool = True,
             label_names: list = None) -> dict:
    """
    Full evaluation: accuracy, F1, ROC-AUC, PR-AUC.
    Returns a metrics dict and prints a report.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro",  zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if binary:
        y_score  = y_proba[:, 1]
        roc_auc  = roc_auc_score(y_test, y_score)
        prauc    = pr_auc_binary(y_test, y_score)
    else:
        classes  = np.unique(y_test)
        roc_auc  = roc_auc_score(
            label_binarize(y_test, classes=classes), y_proba,
            average="macro", multi_class="ovr"
        )
        prauc_d  = pr_auc_multiclass(y_test, y_proba, classes)
        prauc    = prauc_d["macro"]

    metrics = {
        "accuracy":  round(acc,    4),
        "f1_macro":  round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "roc_auc":   round(roc_auc, 4),
        "pr_auc":    round(prauc,  4),
    }

    print("\n" + "="*55)
    print(" EVALUATION RESULTS")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.4f}")
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=label_names if label_names else None,
        zero_division=0
    ))

    return metrics


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          label_names: list = None,
                          title: str = "Confusion Matrix",
                          save_path: str = None) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(np.unique(y_true))),
                                    max(6, len(np.unique(y_true)))))
    sns.heatmap(
        pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=label_names or "auto",
        yticklabels=label_names or "auto",
        ax=ax, cbar_kws={"label": "%"}
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / "confusion_matrix.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")


# ── Plot: Precision-Recall Curve (binary) ────────────────────────────────────
def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray,
                  model_name: str = "Model",
                  save_path: str = None) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where="post", color="#2196F3", lw=2,
            label=f"PR-AUC = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#2196F3", step="post")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--",
               label=f"Baseline = {y_true.mean():.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve: {model_name}", fontweight="bold")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / f"pr_curve_{model_name.replace(' ','_')}.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")


# ── Plot: ROC Curve (binary) ─────────────────────────────────────────────────
def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                   model_name: str = "Model",
                   save_path: str = None) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / f"roc_curve_{model_name.replace(' ','_')}.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")


# ── Plot: Feature Importance (XGBoost / RF) ───────────────────────────────────
def plot_feature_importance(model, feature_names: list,
                            top_n: int = 25,
                            title: str = "Feature Importance",
                            save_path: str = None) -> None:
    # XGBoost
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_estimators_"):
        # VotingClassifier – use first estimator that has importances
        for name, est in model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                title = f"{title} ({name})"
                break
        else:
            print("  No feature_importances_ found in ensemble members.")
            return
    else:
        print("  Model does not expose feature importances.")
        return

    idx  = np.argsort(importances)[-top_n:]
    vals = importances[idx]
    nms  = np.array(feature_names)[idx]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
    ax.barh(nms, vals, color="#4CAF50")
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / "feature_importance.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")


# ── Plot: Class Distribution ──────────────────────────────────────────────────
def plot_class_distribution(y: np.ndarray, label_names: list = None,
                             title: str = "Class Distribution",
                             save_path: str = None) -> None:
    from collections import Counter
    counts  = Counter(y)
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]
    names   = [label_names[c] if label_names and c < len(label_names)
               else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=plt.cm.tab20.colors[:len(classes)])
    ax.set_yscale("log")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count (log scale)")
    ax.set_xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{v:,}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / "class_distribution.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")


# ── Multi-model comparison bar chart ─────────────────────────────────────────
def plot_model_comparison(results: dict,
                          metric: str = "pr_auc",
                          save_path: str = None) -> None:
    """
    results : { model_name: metrics_dict }
    """
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values,
                  color=["#2196F3", "#E91E63", "#4CAF50", "#FF9800"][:len(names)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(f"Model Comparison – {metric.upper().replace('_', ' ')}",
                 fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    sp = save_path or str(PLOTS_DIR / f"model_comparison_{metric}.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {sp}")
