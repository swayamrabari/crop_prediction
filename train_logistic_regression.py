"""Train and evaluate a multinomial logistic regression crop recommender."""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name("Crop_recommendation.csv")
RESULTS_FILE = Path(__file__).with_name("model_results.json")
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET = "label"


def main():
    data = pd.read_csv(DATA_FILE)
    # The provided CSV contains trailing spaces in several header names.
    data.columns = data.columns.str.strip()
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    C=1.0,
                ),
            ),
        ]
    )

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cross_validation,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    )
    print("5-fold cross-validation:")
    for fold, (accuracy, macro_f1) in enumerate(
        zip(scores["test_accuracy"], scores["test_macro_f1"]), start=1
    ):
        print(f"  Fold {fold}: accuracy={accuracy:.4f}, macro F1={macro_f1:.4f}")
    print(
        f"  Mean accuracy: {scores['test_accuracy'].mean():.4f} "
        f"(+/- {scores['test_accuracy'].std():.4f})"
    )
    print(
        f"  Mean macro F1: {scores['test_macro_f1'].mean():.4f} "
        f"(+/- {scores['test_macro_f1'].std():.4f})\n"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    labels = model.named_steps["classifier"].classes_
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    # Coefficients are based on standardized features, making their magnitudes comparable.
    coefficients = model.named_steps["classifier"].coef_
    print("\nMost influential features for each crop:")
    for crop, crop_coefficients in zip(labels, coefficients):
        ranked_features = sorted(
            zip(FEATURES, crop_coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        summary = ", ".join(
            f"{feature} ({coefficient:+.3f})"
            for feature, coefficient in ranked_features[:3]
        )
        print(f"{crop}: {summary}")

    # Example prediction. Replace these values with a new soil/weather observation.
    example = pd.DataFrame(
        [[90, 42, 43, 20.88, 82.00, 6.50, 202.94]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(example)[0]
    best_index = probabilities.argmax()
    print(
        f"\nExample recommendation: {labels[best_index]} "
        f"(probability: {probabilities[best_index]:.2%})"
    )

    classifier = model.named_steps["classifier"]
    results = {
        "dataset": {
            "file": DATA_FILE.name,
            "rows": int(len(data)),
            "features": FEATURES,
            "target": TARGET,
            "classes": classifier.classes_.tolist(),
        },
        "model": {
            "type": "multinomial logistic regression",
            "preprocessing": "StandardScaler",
            "solver": classifier.solver,
            "regularization_strength_C": classifier.C,
            "max_iter": classifier.max_iter,
        },
        "cross_validation": {
            "folds": 5,
            "mean_accuracy": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "mean_macro_f1": float(scores["test_macro_f1"].mean()),
            "macro_f1_std": float(scores["test_macro_f1"].std()),
        },
        "holdout_test": {
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved model information to: {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
