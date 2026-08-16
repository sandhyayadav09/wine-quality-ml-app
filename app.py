"""
app.py
------
My Streamlit web app for the Red Wine Quality classification assignment.

It loads the five models I trained offline, lets the evaluator upload the test
CSV, pick a model from a dropdown, and then shows the six evaluation metrics
plus a confusion matrix and classification report for the chosen model.

Deployed on Streamlit Community Cloud.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

MODEL_DIR = "model"

# Filenames must match what train_models.py wrote out.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")


@st.cache_resource
def load_models():
    """Load all saved models once and keep them in memory."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_feature_names():
    path = os.path.join(MODEL_DIR, "feature_names.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Sidebar: title, description, controls
# ---------------------------------------------------------------------------
st.title("🍷 Red Wine Quality — Classification Explorer")
st.markdown(
    "I trained five classifiers on the **UCI Red Wine Quality** dataset "
    "(binary target: *good* wine when `quality >= 7`). "
    "Upload the test CSV, choose a model, and inspect its performance."
)

models = load_models()
feature_names = load_feature_names()

if not models:
    st.error("No trained models found in the `model/` folder. Run `train_models.py` first.")
    st.stop()

st.sidebar.header("Controls")

# --- Feature (a): CSV upload option ---------------------------------------
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload test_data.csv. It must contain a 'target' column.",
)

# --- Feature (b): model selection dropdown --------------------------------
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()), index=4)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Expected columns: the 15 engineered wine features + a `target` column "
    "(0 = not good, 1 = good)."
)

if uploaded is None:
    st.info("👈 Upload **test_data.csv** from the sidebar to see results.")
    if feature_names:
        with st.expander("Expected feature columns"):
            st.write(feature_names)
    st.stop()

# ---------------------------------------------------------------------------
# Read + validate the uploaded data
# ---------------------------------------------------------------------------
try:
    data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

if "target" not in data.columns:
    st.error("The uploaded CSV must contain a `target` column.")
    st.stop()

X = data.drop(columns=["target"])
y = data["target"]

if feature_names and list(X.columns) != list(feature_names):
    st.warning(
        "Uploaded columns don't exactly match the training features. "
        "I'll reorder/select the known feature columns where possible."
    )
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    X = X[feature_names]

st.subheader("Preview of uploaded test data")
st.dataframe(data.head(), use_container_width=True)
st.caption(f"{data.shape[0]} rows × {data.shape[1]} columns")

# ---------------------------------------------------------------------------
# Run the selected model
# ---------------------------------------------------------------------------
model = models[model_choice]
y_pred = model.predict(X)
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X)[:, 1]
else:
    y_score = model.decision_function(X)

metrics = compute_metrics(y, y_pred, y_score)

# --- Feature (c): display of evaluation metrics ---------------------------
st.subheader(f"Evaluation metrics — {model_choice}")
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.metric(name, f"{val:.3f}")

# --- Feature (d): confusion matrix + classification report ----------------
left, right = st.columns(2)

with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification Report**")
    report = classification_report(
        y, y_pred, target_names=["Not good", "Good"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
# Bonus: compare every model on the uploaded data at a glance
# ---------------------------------------------------------------------------
st.subheader("All models on this test data")
rows = {}
for name, m in models.items():
    p = m.predict(X)
    s = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)
    rows[name] = compute_metrics(y, p, s)
compare_df = pd.DataFrame(rows).T.round(3)
compare_df.index.name = "ML Model Name"
st.dataframe(compare_df, use_container_width=True)
st.bar_chart(compare_df[["Accuracy", "AUC", "F1", "MCC"]])
