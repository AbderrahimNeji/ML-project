"""Iris classifier with train-only scaling and input validation."""

import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_FEATURES = 4
IRIS_TARGET_NAMES = ("setosa", "versicolor", "virginica")
# Physical cm bounds: wider than the sklearn Iris min/max so slightly
# unusual flowers are allowed, while NaN/negatives/metre-scale values fail.
MAX_MEASUREMENT_CM = 20.0


class InvalidIrisFeaturesError(ValueError):
    """Raised when feature vectors are the wrong shape or not plausible cm."""


def validate_iris_features(X):
    """Return a float array of shape (n_samples, 4) or raise.

    Accepts a single row or a batch. Rejects non-numeric values, NaN/Inf,
    the wrong number of columns, and non-positive or implausibly large cm.
    """
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidIrisFeaturesError(
            "Iris features must be numeric sepal/petal measurements in cm"
        ) from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise InvalidIrisFeaturesError(
            f"Iris features must be 1-D or 2-D, got {array.ndim} dimensions"
        )
    if array.shape[1] != N_FEATURES:
        raise InvalidIrisFeaturesError(
            f"Iris features must have {N_FEATURES} values "
            f"(sepal length, sepal width, petal length, petal width); "
            f"got {array.shape[1]}"
        )
    if array.size == 0:
        raise InvalidIrisFeaturesError("Iris features must contain at least one sample")
    if not np.isfinite(array).all():
        raise InvalidIrisFeaturesError("Iris features must be finite (no NaN or Inf)")
    if np.any(array <= 0):
        raise InvalidIrisFeaturesError(
            "Iris measurements are lengths in cm and must be greater than 0"
        )
    if np.any(array > MAX_MEASUREMENT_CM):
        raise InvalidIrisFeaturesError(
            f"Iris measurements above {MAX_MEASUREMENT_CM} cm are not plausible"
        )
    return array


class IrisClassifier:
    def __init__(self, random_state=42):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(random_state=random_state, max_iter=200),
                ),
            ]
        )
        self.is_trained = False

    def _require_trained(self, action):
        if not self.is_trained:
            raise ValueError(f"Model must be trained before {action}")

    def train(self, X_train, y_train):
        """Fit scaler and classifier on training data only."""
        X_train = validate_iris_features(X_train)
        y_train = np.asarray(y_train)
        if len(y_train) != len(X_train):
            raise InvalidIrisFeaturesError(
                f"y_train length {len(y_train)} does not match "
                f"X_train rows {len(X_train)}"
            )
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        """Return class ids after validating and scaling features."""
        self._require_trained("making predictions")
        X = validate_iris_features(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probabilities after validating and scaling features."""
        self._require_trained("making predictions")
        X = validate_iris_features(X)
        return self.model.predict_proba(X)

    def predict_species(self, X):
        """Return one dict per row with species name and class probabilities."""
        X = validate_iris_features(X)
        labels = self.predict(X)
        probas = self.predict_proba(X)
        results = []
        for label, proba in zip(labels, probas):
            class_id = int(label)
            results.append(
                {
                    "class_id": class_id,
                    "species": IRIS_TARGET_NAMES[class_id],
                    "probabilities": {
                        name: float(p)
                        for name, p in zip(IRIS_TARGET_NAMES, proba)
                    },
                }
            )
        return results

    def evaluate(self, X_test, y_test):
        """Score held-out data with the train-fitted scaler (no refit)."""
        self._require_trained("evaluation")
        X_test = validate_iris_features(X_test)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=IRIS_TARGET_NAMES)
        return accuracy, report

    def scaler_mean(self):
        """Training-set feature means stored by StandardScaler."""
        self._require_trained("reading scaler statistics")
        return self.model.named_steps["scaler"].mean_

    def feature_importance(self):
        """Mean |coefficient| across classes (multinomial logistic regression)."""
        self._require_trained("reading feature importance")
        coef = self.model.named_steps["clf"].coef_
        return np.mean(np.abs(coef), axis=0)

    def save_model(self, filepath="models/iris_classifier.pkl"):
        """Save the fitted scaler+classifier pipeline."""
        self._require_trained("saving")
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, filepath)

    def load_model(self, filepath="models/iris_classifier.pkl"):
        """Load a pipeline (or a legacy bare estimator)."""
        loaded = joblib.load(filepath)
        if isinstance(loaded, Pipeline):
            self.model = loaded
        elif hasattr(loaded, "predict"):
            # Older pickles stored LogisticRegression without a scaler.
            self.model = loaded
        else:
            raise ValueError(f"Unsupported model file: {filepath}")
        self.is_trained = True
