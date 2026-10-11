import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, target_names=None):
    """Plot confusion matrix"""
    if target_names is None:
        iris = load_iris()
        target_names = iris.target_names

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def _estimator_coefficients(model):
    """Return (n_classes, n_features) coefficients from a pipeline or estimator."""
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model)
    if not hasattr(estimator, "coef_"):
        raise AttributeError("Model does not expose logistic regression coefficients")
    return np.asarray(estimator.coef_)


def plot_feature_importance(model, feature_names):
    """Plot mean |coefficient| across classes for logistic regression."""
    importance = np.mean(np.abs(_estimator_coefficients(model)), axis=0)
    feature_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp["feature"], feature_imp["importance"])
    plt.title("Feature Importance (Logistic Regression)")
    plt.xlabel("Mean Absolute Coefficient Value")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
