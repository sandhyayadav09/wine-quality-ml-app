"""
train_models.py
----------------
I use this script to prepare the Red Wine Quality dataset, engineer a few extra
features so my feature count clears the assignment's minimum of 12, train the
five required classifiers, compute all six evaluation metrics for each, and save
the trained pipelines so my Streamlit app can load them without retraining.

Run once locally:  python train_models.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

RANDOM_STATE = 42
MODEL_DIR = "model"
RAW_FILE = "winequality-red.csv"


def load_and_prepare():
    """Load the raw UCI file and turn it into a supervised binary problem."""
    # The UCI file is semicolon separated.
    df = pd.read_csv(RAW_FILE, sep=";")

    # --- Feature engineering -------------------------------------------------
    # The raw file has 11 chemical features. The assignment needs at least 12,
    # so I add a handful of chemistry-motivated ratios/interactions that a wine
    # analyst would actually reason about. These are original combinations, not
    # copied columns.
    df["free_to_total_so2"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    df["bound_so2"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]
    df["acidity_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-6)

    # --- Target --------------------------------------------------------------
    # I convert the 0-10 quality score into a binary label: a wine is "good"
    # (1) when quality >= 7, otherwise "not good" (0). This is the standard
    # binarisation used for this dataset and gives a clean 2-class problem.
    df["target"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])

    X = df.drop(columns=["target"])
    y = df["target"]
    return df, X, y


def build_models():
    """Return the five classifiers the assignment asks for.

    Logistic Regression, kNN and Naive Bayes are scale sensitive, so I wrap
    them in a StandardScaler pipeline. Trees / forests are scale invariant, so
    I leave them unscaled.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def evaluate(model, X_test, y_test):
    """Compute the six required metrics for a fitted model."""
    y_pred = model.predict(X_test)

    # AUC needs a probability / score for the positive class.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, X, y = load_and_prepare()

    print(f"Rows: {len(df)}  |  Features: {X.shape[1]}  |  "
          f"Positive class (good wine): {int(y.sum())} "
          f"({y.mean() * 100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # I save the held-out test split as the CSV the assignment wants uploaded
    # to the Streamlit app. It keeps the deployed app light (free-tier friendly).
    test_out = X_test.copy()
    test_out["target"] = y_test.values
    test_out.to_csv("test_data.csv", index=False)
    print("Saved test_data.csv:", test_out.shape)

    # Also save the exact feature order so the app can validate uploads.
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

    models = build_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"Trained + saved: {name}")

    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"))

    print("\n=== Comparison Table ===")
    print(metrics_df.to_string())


if __name__ == "__main__":
    main()
