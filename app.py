"""Streamlit front-end for comparing five classifiers on the UCI handwritten
digits dataset.

The app loads the pipelines persisted by ``model/train_models.py`` and lets a
user upload a hold-out CSV (or fall back to the bundled ``test_data.csv``) to
inspect per-model evaluation metrics, a confusion matrix, a classification
report and a side-by-side comparison across all models.

Run locally:  ``streamlit run app.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
AVERAGE = "macro"

st.set_page_config(
    page_title="Handwritten Digit Classifier Studio",
    page_icon="✏️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    meta = load_metadata()
    return {key: joblib.load(MODEL_DIR / f"{key}.pkl") for key in meta["models"]}


@st.cache_data(show_spinner=False)
def load_bundled_test_data() -> pd.DataFrame | None:
    path = APP_DIR / "test_data.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_metrics(pipeline, X, y_true, class_labels: list[int]) -> dict[str, float]:
    """Return the six evaluation metrics required by the assignment."""
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


def coerce_labels(series: pd.Series, class_labels: list[int]) -> pd.Series:
    """Best-effort cast of the uploaded label column to the trained dtype."""
    if all(isinstance(c, (int, np.integer)) for c in class_labels):
        try:
            return series.astype(int)
        except (ValueError, TypeError):
            return series
    return series


def render_digit_gallery(X: pd.DataFrame, y_true, y_pred, n: int = 8) -> None:
    """Show a few 8x8 digit thumbnails with their true / predicted labels."""
    n = min(n, len(X))
    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 2.0))
    for ax, (_, row), true_label, pred_label in zip(
        np.atleast_1d(axes), X.head(n).iterrows(), y_true[:n], y_pred[:n]
    ):
        ax.imshow(row.to_numpy().reshape(8, 8), cmap="gray_r")
        correct = int(true_label) == int(pred_label)
        ax.set_title(
            f"true {true_label}\npred {pred_label}",
            fontsize=9,
            color="green" if correct else "red",
        )
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
meta = load_metadata()
models = load_models()
feature_names: list[str] = meta["feature_names"]
target_column: str = meta["target_column"]
class_labels: list[int] = meta["class_labels"]
display_names: dict[str, str] = meta["models"]

st.title("✏️ Handwritten Digit Classifier Studio")
st.caption(
    "Compare Logistic Regression, Decision Tree, kNN, Naive Bayes and a "
    "Random Forest ensemble on the UCI *Optical Recognition of Handwritten "
    "Digits* dataset (64 pixel features, 10 classes)."
)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Dataset")
    st.markdown(
        f"- **Source:** {meta['dataset']}\n"
        f"- **Task:** {meta['task_type']}\n"
        f"- **Features:** {len(feature_names)}\n"
        f"- **Classes:** {len(class_labels)} (digits {class_labels[0]}–{class_labels[-1]})"
    )

    st.subheader("Model")
    selected_key = st.selectbox(
        "Choose a classification model",
        options=list(display_names.keys()),
        format_func=lambda k: display_names[k],
    )

    st.subheader("Test data (CSV)")
    uploaded = st.file_uploader(
        "Upload a test CSV (features + label column). "
        "Leave empty to use the bundled test_data.csv.",
        type=["csv"],
    )

# ---- Resolve the test dataframe -------------------------------------------
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    data_source = "uploaded file"
else:
    test_df = load_bundled_test_data()
    data_source = "bundled test_data.csv"

if test_df is None:
    st.error("No test data available. Upload a CSV to continue.")
    st.stop()

missing = [c for c in feature_names if c not in test_df.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required feature column(s), "
        f"e.g. {missing[:5]}. Please upload data with the same columns as "
        "test_data.csv."
    )
    st.stop()

# Choose the label column (default to the trained target column when present).
non_feature_cols = [c for c in test_df.columns if c not in feature_names]
if target_column in test_df.columns:
    default_label = target_column
elif non_feature_cols:
    default_label = non_feature_cols[-1]
else:
    default_label = None

label_col = st.sidebar.selectbox(
    "Label column",
    options=non_feature_cols or ["<none>"],
    index=(non_feature_cols.index(default_label) if default_label in non_feature_cols else 0),
    help="Column holding the ground-truth class. Needed to compute metrics.",
)

X = test_df[feature_names]
has_labels = label_col in test_df.columns
y_true = coerce_labels(test_df[label_col], class_labels) if has_labels else None

# ---- Data overview ---------------------------------------------------------
st.subheader("1 · Test data overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(test_df):,}")
c2.metric("Feature columns", len(feature_names))
c3.metric("Source", data_source)

with st.expander("Preview the first rows"):
    st.dataframe(test_df.head(10), width="stretch")

if has_labels:
    dist = y_true.value_counts().sort_index()
    st.bar_chart(dist, x_label="digit class", y_label="count")

if not has_labels:
    st.warning(
        "No label column selected, so evaluation metrics cannot be computed. "
        "Showing model predictions only."
    )
    preds = models[selected_key].predict(X)
    st.dataframe(
        pd.DataFrame({"row": np.arange(len(preds)), "prediction": preds}),
        width="stretch",
    )
    st.stop()

# ---- Selected-model metrics ------------------------------------------------
pipeline = models[selected_key]
y_pred = pipeline.predict(X)
metrics = compute_metrics(pipeline, X, y_true, class_labels)

st.subheader(f"2 · Evaluation metrics — {display_names[selected_key]}")
m_cols = st.columns(6)
for col, (name, value) in zip(m_cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

# ---- Confusion matrix + classification report ------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{display_names[selected_key]} — confusion matrix")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, width="stretch")

# ---- Sample predictions gallery -------------------------------------------
st.markdown("**Sample predictions** (green = correct, red = misclassified)")
render_digit_gallery(X, y_true.to_numpy(), y_pred)

# ---- Model comparison ------------------------------------------------------
st.subheader("3 · Model comparison on this test data")
comparison = {
    display_names[key]: compute_metrics(model, X, y_true, class_labels)
    for key, model in models.items()
}
comparison_df = pd.DataFrame(comparison).T[
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]

st.dataframe(
    comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"),
    width="stretch",
)

best_model = comparison_df["Accuracy"].idxmax()
st.success(
    f"🏆 Best accuracy on this test data: **{best_model}** "
    f"({comparison_df.loc[best_model, 'Accuracy']:.4f})"
)

st.markdown("**Accuracy vs F1 across models**")
st.bar_chart(comparison_df[["Accuracy", "F1"]])

st.caption(
    "Metrics use macro averaging; AUC uses a one-vs-rest scheme for this "
    "multi-class problem. Built with scikit-learn + Streamlit."
)
