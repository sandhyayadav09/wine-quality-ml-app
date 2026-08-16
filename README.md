# 🍷 Red Wine Quality — Multi-Model Classification & Streamlit App

Machine Learning Assignment 2 — M.Tech (AIML/DSE), BITS Pilani WILP.

**Name:** Sandhya Yadav  |  **BITS ID:** 2025ac05593

I implement five classification models on a single dataset, evaluate each of
them on six metrics, and expose the results through an interactive Streamlit web
app deployed on Streamlit Community Cloud.

---

## a. Problem statement

I want to predict whether a red wine is **"good"** based purely on its physico-
chemical measurements (acidity, sugar, sulphates, alcohol, etc.). I frame this
as a **binary classification** problem: a wine is labelled **good (1)** when its
sensory `quality` score is **7 or higher**, and **not good (0)** otherwise.

This matters because sensory scoring by human tasters is slow and subjective.
If a model can flag likely high-quality wines from lab measurements alone, a
producer can prioritise which batches to send for expert tasting. My goal is to
compare several standard classifiers and identify which one predicts wine
quality most reliably on this dataset.

---

## b. Dataset description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Wine Quality* (red) |
| Link | https://archive.ics.uci.edu/dataset/186/wine+quality |
| Raw file | `winequality-red.csv` (semicolon-separated) |
| Instances | **1,599** (≥ 500 required ✔) |
| Original features | 11 physico-chemical measurements |
| Engineered features | 4 (see below) → **15 features total** (≥ 12 required ✔) |
| Target | `target` — binary: 1 = good (`quality ≥ 7`), 0 = not good |
| Class balance | 217 good (13.6%) vs 1,382 not good (86.4%) — **imbalanced** |

**Original 11 features:** fixed acidity, volatile acidity, citric acid,
residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density,
pH, sulphates, alcohol.

**4 engineered features** (added so the feature count meets the ≥ 12 minimum;
each is chemistry-motivated, not a duplicate column):

| New feature | Definition | Reasoning |
|---|---|---|
| `free_to_total_so2` | free SO₂ / total SO₂ | share of *active* (antimicrobial) sulphur dioxide |
| `bound_so2` | total SO₂ − free SO₂ | the inactive, already-reacted SO₂ |
| `acidity_ratio` | fixed acidity / volatile acidity | balance of pleasant vs vinegary acidity |
| `sugar_alcohol_ratio` | residual sugar / alcohol | rough sweetness-to-body indicator |

The **class imbalance** (only ~14% good wines) is the key modelling challenge:
plain accuracy is misleading here (a model predicting "not good" for everything
scores ~86%), which is exactly why AUC, F1 and MCC are the metrics I trust most.

---

## c. GitHub repository link

> **Repo:** https://github.com/sandhyayadav09/wine-quality-ml-app

**Live Streamlit app:** `https://<your-app-name>.streamlit.app`
_(replace with the URL Streamlit gives you after deploying)_

### Repository structure

```
wine-quality-ml-app/
├── app.py                 # Streamlit web application
├── train_models.py        # trains all 5 models + computes metrics
├── requirements.txt       # pinned dependencies (deployment-safe)
├── README.md              # this file
├── test_data.csv          # held-out test split, uploaded in the app
├── winequality-red.csv    # raw UCI dataset (for reproducibility)
└── model/                 # saved artefacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── feature_names.pkl
    └── metrics.csv
```

---

## d. Models used

I implement the following classifiers, all on the **same** dataset and the same
80/20 stratified train/test split (`random_state=42`):

1. **Logistic Regression** (scaled)
2. **Decision Tree** (`max_depth=8`)
3. **k-Nearest Neighbours** (`k=15`, scaled)
4. **Naive Bayes** (Gaussian, scaled)
5. **Random Forest** (300 trees, `class_weight="balanced"`) — ensemble model

Scale-sensitive models (Logistic Regression, kNN, Naive Bayes) are wrapped in a
`StandardScaler` pipeline; tree-based models are left unscaled.

### Comparison table (on the 20% held-out test set, 320 wines)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8906 | 0.8941 | 0.6818 | 0.3488 | 0.4615 | 0.4361 |
| Decision Tree | 0.8906 | 0.8349 | 0.5909 | 0.6047 | 0.5977 | 0.5345 |
| kNN | 0.9094 | 0.8619 | 0.8182 | 0.4186 | 0.5538 | 0.5448 |
| Naive Bayes | 0.8125 | 0.8436 | 0.3896 | **0.6977** | 0.5000 | 0.4213 |
| **Random Forest** | **0.9406** | **0.9375** | **1.0000** | 0.5581 | **0.7164** | **0.7227** |

_(Best value in each column shown in bold; numbers are reproducible by running
`python train_models.py`.)_

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | High accuracy (0.89) but that is inflated by the imbalance — its **recall is only 0.35**, so it misses ~65% of the truly good wines. As a linear model it can't capture the non-linear chemistry, though its AUC (0.89) shows it ranks wines reasonably. |
| **Decision Tree** | Best **recall (0.60)** among the simpler models and a balanced F1 (0.60), because a single tree can carve non-linear regions. But it has the **lowest AUC (0.83)** — a single hard-split tree ranks probabilities poorly and tends to overfit. |
| **kNN** | Highest **precision (0.82)** of all models: when it says "good", it is usually right. But recall is low (0.42), so it is conservative and misses many good wines. Sensitive to scaling (hence the StandardScaler) and to the imbalance. |
| **Naive Bayes** | Highest **recall (0.70)** — it flags the most good wines — but pays for it with the **lowest precision (0.39)** and lowest accuracy. Its feature-independence assumption is unrealistic here (many wine chemistry features are correlated), so its predictions are noisy. |
| **Random Forest** | **Best on 5 of 6 metrics** — top Accuracy (0.94), AUC (0.94), Precision (1.00), F1 (0.72) and MCC (0.72). Its recall (0.56) is mid-pack, but a **precision of 1.00** means every wine it labels "good" really is good — no false alarms. The ensemble of de-correlated trees plus `class_weight="balanced"` handles both the non-linearity and the imbalance, giving the best overall trade-off. |
| **Overall Winner for my dataset?** | **Random Forest.** On an imbalanced dataset, MCC and AUC are the fairest single-number summaries, and Random Forest leads both by a clear margin (MCC 0.72 vs the next-best 0.54; AUC 0.94 vs 0.89). It combines the highest accuracy, F1 and a perfect precision, making it the most trustworthy classifier for flagging genuinely good wines. |

---

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # trains models + writes test_data.csv + model/
streamlit run app.py        # opens the interactive app
```

## Streamlit app features

- **CSV upload** — upload `test_data.csv` (only test data, to stay within the
  free tier).
- **Model selection dropdown** — switch between all five trained models.
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC shown live.
- **Confusion matrix + classification report** for the selected model.
- Bonus: an all-models comparison table and chart on the uploaded data.

---

## Notes on scope

The assignment text says "6 ML models" but lists **five** distinct algorithms in
Step 2, and the comparison/observation tables have five model rows. I therefore
implement the five listed models (Logistic Regression, Decision Tree, kNN,
Naive Bayes, Random Forest ensemble) exactly as tabulated.
