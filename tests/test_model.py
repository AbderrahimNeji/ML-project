import os
import sys

import numpy as np
import pytest

from src.data_loader import load_iris_data
from src.model import (
    IRIS_TARGET_NAMES,
    InvalidIrisFeaturesError,
    IrisClassifier,
    validate_iris_features,
)
from src.utils import _estimator_coefficients

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestIrisClassifier:
    def setup_method(self):
        """Setup method that runs before each test"""
        self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data(
            test_size=0.3, random_state=42
        )
        self.classifier = IrisClassifier()

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        assert not self.classifier.is_trained
        assert self.classifier.model is not None

    def test_model_training(self):
        """Test model training functionality"""
        self.classifier.train(self.X_train, self.y_train)
        assert self.classifier.is_trained

    def test_model_prediction(self):
        """Test model prediction functionality"""
        self.classifier.train(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test[:5])
        assert len(predictions) == 5
        assert all(isinstance(pred, (np.int32, np.int64, int)) for pred in predictions)

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        self.classifier.train(self.X_train, self.y_train)
        accuracy, report = self.classifier.evaluate(self.X_test, self.y_test)
        assert 0 <= accuracy <= 1
        assert isinstance(report, str)
        assert "precision" in report.lower()

    def test_model_save_load(self, tmp_path):
        """Test model saving and loading"""
        self.classifier.train(self.X_train, self.y_train)

        save_path = tmp_path / "test_model.pkl"
        self.classifier.save_model(str(save_path))
        assert save_path.exists()

        new_classifier = IrisClassifier()
        new_classifier.load_model(str(save_path))
        assert new_classifier.is_trained

        original_pred = self.classifier.predict(self.X_test[:5])
        loaded_pred = new_classifier.predict(self.X_test[:5])
        assert np.array_equal(original_pred, loaded_pred)

    def test_scaler_fits_training_split_only(self):
        """StandardScaler must use train means, not the held-out test split."""
        self.classifier.train(self.X_train, self.y_train)
        scaler_mean = self.classifier.scaler_mean()
        np.testing.assert_allclose(scaler_mean, self.X_train.mean(axis=0))
        assert not np.allclose(scaler_mean, self.X_test.mean(axis=0))

    def test_known_setosa_predicts_setosa(self):
        self.classifier.train(self.X_train, self.y_train)
        result = self.classifier.predict_species([5.1, 3.5, 1.4, 0.2])[0]
        assert result["species"] == "setosa"
        assert result["class_id"] == 0
        setosa_p = result["probabilities"]["setosa"]
        assert setosa_p == max(result["probabilities"].values())
        assert pytest.approx(sum(result["probabilities"].values()), rel=1e-6) == 1.0

    def test_predict_rejects_untrained_model(self):
        with pytest.raises(ValueError, match="trained before making predictions"):
            self.classifier.predict([5.1, 3.5, 1.4, 0.2])

    def test_feature_importance_uses_all_classes(self):
        self.classifier.train(self.X_train, self.y_train)
        importance = self.classifier.feature_importance()
        coef = _estimator_coefficients(self.classifier.model)
        assert coef.shape == (3, 4)
        np.testing.assert_allclose(importance, np.mean(np.abs(coef), axis=0))


def test_data_loading():
    """Test data loading functionality"""
    X_train, X_test, y_train, y_test = load_iris_data()
    assert X_train.shape[1] == 4  # 4 features
    assert len(np.unique(y_train)) == 3  # 3 classes
    assert len(X_train) + len(X_test) == 150  # Total samples


def test_validate_accepts_single_row_and_batch():
    row = validate_iris_features([5.1, 3.5, 1.4, 0.2])
    assert row.shape == (1, 4)
    batch = validate_iris_features([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    assert batch.shape == (2, 4)


@pytest.mark.parametrize(
    "features,match",
    [
        ([5.1, 3.5, 1.4], "must have 4 values"),
        ([5.1, 3.5, 1.4, 0.2, 1.0], "must have 4 values"),
        ([5.1, 3.5, 1.4, float("nan")], "finite"),
        ([5.1, 3.5, 1.4, float("inf")], "finite"),
        ([5.1, 3.5, 1.4, -0.2], "greater than 0"),
        ([5.1, 3.5, 1.4, 0.0], "greater than 0"),
        ([5.1, 3.5, 1.4, 50.0], "not plausible"),
        ("bad", "numeric"),
    ],
)
def test_validate_rejects_invalid_features(features, match):
    with pytest.raises(InvalidIrisFeaturesError, match=match):
        validate_iris_features(features)


def test_parse_cli_features_and_format_prediction():
    from src.predict import format_prediction, parse_cli_features

    parsed = parse_cli_features(["5.1", "3.5", "1.4", "0.2"])
    np.testing.assert_allclose(parsed, [[5.1, 3.5, 1.4, 0.2]])
    assert parse_cli_features([]) is None
    with pytest.raises(InvalidIrisFeaturesError, match="usage: predict.py"):
        parse_cli_features(["5.1", "3.5"])
    with pytest.raises(InvalidIrisFeaturesError, match="must be numbers"):
        parse_cli_features(["5.1", "3.5", "1.4", "wide"])

    text = format_prediction(
        {
            "species": "setosa",
            "probabilities": {
                name: 1.0 if name == "setosa" else 0.0
                for name in IRIS_TARGET_NAMES
            },
        }
    )
    assert "Prediction: setosa" in text
    assert "setosa: 1.0000" in text
