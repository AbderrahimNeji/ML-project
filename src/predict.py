#!/usr/bin/env python3
"""
Prediction script for Iris classifier
"""

import sys

try:
    from src.model import (
        IRIS_TARGET_NAMES,
        InvalidIrisFeaturesError,
        IrisClassifier,
        validate_iris_features,
    )
except ImportError:  # `python src/predict.py` puts src/ on sys.path
    from model import (
        IRIS_TARGET_NAMES,
        InvalidIrisFeaturesError,
        IrisClassifier,
        validate_iris_features,
    )


DEFAULT_EXAMPLES = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [6.7, 3.0, 5.2, 2.3],  # Virginica
    [5.9, 3.0, 4.2, 1.5],  # Versicolor
]


def parse_cli_features(argv):
    """Parse four measurement arguments, or return None to use examples."""
    if not argv:
        return None
    if len(argv) != 4:
        raise InvalidIrisFeaturesError(
            "usage: predict.py sepal_length sepal_width petal_length petal_width"
        )
    try:
        values = [float(item) for item in argv]
    except ValueError as exc:
        raise InvalidIrisFeaturesError(
            "all four measurements must be numbers (cm)"
        ) from exc
    return validate_iris_features(values)


def format_prediction(result):
    lines = [
        f"Prediction: {result['species']}",
        "Probabilities:",
    ]
    for name in IRIS_TARGET_NAMES:
        lines.append(f"  {name}: {result['probabilities'][name]:.4f}")
    return "\n".join(lines)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    print("Iris Classifier Prediction")

    classifier = IrisClassifier()
    try:
        classifier.load_model("models/iris_classifier.pkl")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
        return 1

    try:
        requested = parse_cli_features(argv)
        samples = requested if requested is not None else validate_iris_features(
            DEFAULT_EXAMPLES
        )
        results = classifier.predict_species(samples)
    except InvalidIrisFeaturesError as exc:
        print(f"Invalid input: {exc}")
        return 1

    print("\nPredictions:")
    print("Features: [sepal length, sepal width, petal length, petal width]")
    for i, (features, result) in enumerate(zip(samples, results), 1):
        print(f"\nExample {i}: {features.tolist()}")
        print(format_prediction(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
