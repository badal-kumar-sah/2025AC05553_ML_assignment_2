# Handwritten Digit Classifier Studio 🖊️

An end-to-end machine-learning project that trains, evaluates and compares **five
classification models** on the UCI *Optical Recognition of Handwritten Digits*
dataset, and serves them through an interactive **Streamlit** web app.

---

## a. Problem statement

Given an 8×8 grayscale scan of a single handwritten digit (flattened into 64
pixel-intensity features), predict which digit **0–9** was written. This is a
**multi-class classification** problem (10 balanced classes) that underpins
real-world optical character recognition (OCR) tasks such as reading postal
codes and processing bank cheques.

The goal of this assignment is to implement several classical classifiers on the
**same** dataset, compare them across six evaluation metrics, and expose the
results through a deployed, interactive front-end where a user can upload test
data and inspect each model's behaviour.

## b. Dataset description

| Property | Value |
| --- | --- |
| Name | Optical Recognition of Handwritten Digits |
| Source | UCI Machine Learning Repository (bundled with scikit-learn as `load_digits`) |
| Instances | **1,797** (≥ 500 required) |
| Features | **64** numeric pixel intensities `pixel_0_0 … pixel_7_7`, each in `[0, 16]` (≥ 12 required) |
| Target | `digit_class` — the digit label `0–9` |
| Classes | 10 (balanced, ~180 samples each) |
| Task | Multi-class classification |
| Split | 80% train / 20% test, stratified, `random_state=42` (360 test rows) |

Each row is a low-resolution 8×8 bitmap of a handwritten digit. Pixel values are
the number of "on" pixels aggregated from the original 32×32 NIST bitmaps, which
mildly blurs the images and reduces the impact of small distortions.

The 20% stratified hold-out split is exported to [`test_data.csv`](test_data.csv)
and is the file the Streamlit app consumes.

## c. GitHub repository link

> **➡️ Replace with your repository URL after pushing:**
> `https://github.com/<your-username>/handwritten-digit-classifier-studio`

The repository contains the complete source code, `requirements.txt`, this
`README.md`, the test data (`test_data.csv`), and the `model/` folder with the
training script and all five persisted models.

## d. Models used

All five models are trained on the **same** dataset and split. Distance- and
gradient-based learners (Logistic Regression, kNN) are wrapped in a
`StandardScaler`; tree-based learners use the raw pixels. Metrics are computed on
the 360-row hold-out test set. `Precision`, `Recall` and `F1` use **macro**
averaging and `AUC` uses a **one-vs-rest** scheme (appropriate for this
multi-class problem).

### Comparison table (hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9722 | 0.9991 | 0.9721 | 0.9719 | 0.9719 | 0.9692 |
| Decision Tree | 0.8194 | 0.8995 | 0.8226 | 0.8188 | 0.8186 | 0.7998 |
| kNN | 0.9639 | 0.9950 | 0.9646 | 0.9637 | 0.9634 | 0.9600 |
| Naive Bayes | 0.8111 | 0.9705 | 0.8463 | 0.8103 | 0.8137 | 0.7940 |
| Random Forest (Ensemble) | 0.9694 | 0.9992 | 0.9700 | 0.9691 | 0.9689 | 0.9662 |

*(These exact numbers are reproducible by running `python model/train_models.py`
and are also shown live in the app when `test_data.csv` is loaded.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | Best overall — highest accuracy (0.9722) and F1 (0.9719). After standardization the 64 pixel features are close to linearly separable across the digits, so a regularized linear model generalizes strongly with a near-perfect AUC (0.9991). Also the smallest, fastest model to serve. |
| **Decision Tree** | Weakest of the group (accuracy 0.8194). A single axis-aligned tree splits on individual pixel thresholds and cannot capture the smooth spatial correlation between neighbouring pixels; capping depth at 12 controls variance but the model still misclassifies visually similar digits. |
| **kNN** | Very strong (0.9639) once features are scaled — visually similar digits sit close together in 64-D Euclidean space, so majority voting over 5 neighbours works well. Slightly below the top two, and it is a lazy learner that stores the whole training set (largest pickle, slowest inference). |
| **Naive Bayes** | Lowest accuracy (0.8111) because the feature-independence assumption is clearly violated — adjacent pixels are highly correlated. Interestingly its AUC (0.9705) and precision (0.8463) remain high, so its probability *ranking* is still useful even when the hard prediction is wrong. |
| **Random Forest (Ensemble)** | Essentially tied for first (accuracy 0.9694, and the best AUC 0.9992). Bagging 300 de-correlated trees removes the single tree's overfitting and captures pixel interactions, at the cost of the largest model file and slower training. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it leads on accuracy, precision, recall, F1 and MCC, with **Random Forest** a very close second (and a marginally higher AUC). For a lightweight, fast and highly accurate deployment, Logistic Regression is the recommended model for this dataset. |

---

## Streamlit app features

The deployed app ([`app.py`](app.py)) implements every required feature:

1. **CSV upload** — upload a test CSV; if none is provided it falls back to the
   bundled `test_data.csv`.
2. **Model selection dropdown** — switch between all five trained models.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC shown as
   metric cards for the selected model.
4. **Confusion matrix + classification report** — a Seaborn heatmap plus a
   per-class precision/recall/F1 table, with a gallery of sample digit
   predictions (green = correct, red = misclassified).
5. **Model comparison** — a live table (best value per metric highlighted) and an
   Accuracy-vs-F1 bar chart across all five models on the uploaded test data.

## Project structure

```
ML_assignment/
├── app.py                 # Streamlit front-end
├── requirements.txt       # Pinned dependencies (Streamlit Cloud compatible)
├── README.md              # This file
├── test_data.csv          # Hold-out test split (upload this in the app)
└── model/
    ├── train_models.py    # Trains + evaluates + persists all models
    ├── metadata.json      # Feature/target names, class labels, metric table
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## Run locally (with `uv`)

```bash
# 1. Create an environment and install dependencies
uv venv --python 3.11
uv pip install -r requirements.txt

# 2. (Optional) retrain the models and regenerate test_data.csv
uv run python model/train_models.py

# 3. Launch the app
uv run streamlit run app.py
```

Then open the local URL shown in the terminal (default http://localhost:8501)
and upload `test_data.csv`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, main file `app.py`.
4. Click **Deploy**. The pinned `requirements.txt` reproduces the training
   environment so the persisted models load without version warnings.

> **➡️ Live app URL (fill in after deploying):**
> `https://<your-app-name>.streamlit.app`

## Reproducibility & notes

- Every run uses `random_state=42` for the split and the models, so results are
  deterministic.
- `requirements.txt` pins the exact versions used to train the models, which
  guarantees the `joblib` pickles load correctly on Streamlit Cloud.
- The BITS Virtual Lab execution screenshot is included in the submission PDF.
