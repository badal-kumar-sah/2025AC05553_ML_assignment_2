"""Train, evaluate and persist the classification models used by the app.

Dataset: UCI "Optical Recognition of Handwritten Digits" (bundled with
scikit-learn as ``load_digits``). 1797 samples, 64 numeric pixel features,
10 balanced classes (digits 0-9) -> a multi-class classification problem.

Running this script (``python model/train_models.py``) regenerates every
artifact consumed by ``app.py``:

* ``model/<key>.pkl``    -> one persisted scikit-learn Pipeline per model
* ``model/metadata.json``-> feature / target names, class labels, metric table
* ``test_data.csv``      -> the hold-out test split uploaded to the app
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TARGET_COLUMN = "digit_class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
AVERAGE = "macro"  # equal weight per class -> fair on the balanced digit set


def build_model_specs() -> dict[str, tuple[str, Pipeline]]:
    """Map an internal key -> (display name, untrained pipeline).

    Distance/gradient based learners (Logistic Regression, kNN) are wrapped
    with ``StandardScaler``; tree based learners are scale invariant so they
    are left on the raw pixel intensities.
    """
    return {
        "logistic_regression": (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        "decision_tree": (
            "Decision Tree",
            Pipeline(
                [("clf", DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE))]
            ),
        ),
        "knn": (
            "kNN",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
        "naive_bayes": (
            "Naive Bayes",
            Pipeline([("clf", GaussianNB())]),
        ),
        "random_forest": (
            "Random Forest (Ensemble)",
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                        ),
                    )
                ]
            ),
        ),
    }


def evaluate(pipeline: Pipeline, X, y_true, class_labels: list[int]) -> dict[str, float]:
    """Return the six required metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    if len(class_labels) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average=AVERAGE, labels=class_labels
        )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    bunch = load_digits(as_frame=True)
    features: pd.DataFrame = bunch.data.copy()
    feature_names = list(features.columns)
    target = bunch.target.astype(int)
    class_labels = sorted(int(c) for c in np.unique(target))

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    specs = build_model_specs()
    metrics_table: dict[str, dict[str, float]] = {}

    for key, (display_name, pipeline) in specs.items():
        pipeline.fit(X_train, y_train)
        metrics_table[display_name] = evaluate(pipeline, X_test, y_test, class_labels)
        joblib.dump(pipeline, HERE / f"{key}.pkl")
        print(f"[saved] model/{key}.pkl")

    # Persist the hold-out split as the CSV students upload to the app.
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.to_numpy()
    test_df.to_csv(REPO_ROOT / "test_data.csv", index=False)
    print(f"[saved] test_data.csv  ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")

    metadata = {
        "dataset": "UCI Optical Recognition of Handwritten Digits",
        "task_type": "multiclass",
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,
        "models": {key: display for key, (display, _) in specs.items()},
        "metrics": metrics_table,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "average": AVERAGE,
        "random_state": RANDOM_STATE,
    }
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("[saved] model/metadata.json")

    print("\nHold-out test metrics")
    print(pd.DataFrame(metrics_table).T.round(4).to_string())


if __name__ == "__main__":
    main()
