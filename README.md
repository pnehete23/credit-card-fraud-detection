# Credit Card Fraud Detection — End-to-End ML/DL Pipeline

**Team:** Nadezhda Shiroglazova, Prathamesh Nehete, Sheng Hu  
**Course:** MSDS 422 — Practical Machine Learning  
**Methodology:** CRISP-DM  

## Problem Statement

Credit card fraud causes billions in annual losses globally. This project addresses detecting fraudulent transactions in a highly imbalanced dataset (0.172% fraud rate) using machine learning and deep learning techniques, with emphasis on realistic evaluation protocols.

## Dataset

**Kaggle Credit Card Fraud Detection Dataset**  
- 284,807 transactions from European cardholders (September 2013)  
- 492 fraudulent transactions (0.172%)  
- 28 PCA-transformed features (V1–V28) + Time + Amount  
- Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

## Models Implemented

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | Logistic Regression | ML (Linear) | Baseline classifier with L2 regularization |
| 2 | Random Forest | ML (Ensemble) | Bagging with balanced subsample weighting |
| 3 | XGBoost | ML (Boosting) | Gradient boosting with scale_pos_weight |
| 4 | Deep Autoencoder | DL (Unsupervised) | Anomaly detection via reconstruction error |
| 5 | Neural Network | DL (Supervised) | Feed-forward classifier with batch norm & dropout |

## Resampling Strategies

- **No Resampling** — Baseline
- **Random Undersampling** — Reduce majority class
- **SMOTE** — Synthetic Minority Oversampling
- **Hybrid (SMOTE + Tomek Links)** — Best of both (recommended)

## Project Structure

```
├── credit_card_fraud_detection.py   # Full end-to-end pipeline
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── creditcard.csv                    # Dataset (download from Kaggle)
└── output/                           # Generated artifacts
    ├── 01_class_distribution.png
    ├── 02_amount_distribution.png
    ├── ...
    ├── 17_error_analysis.png
    ├── model_comparison.csv
    ├── resampling_comparison.csv
    ├── xgboost_model.pkl
    ├── random_forest_model.pkl
    ├── logistic_regression_model.pkl
    ├── autoencoder_model.keras
    ├── neural_network_model.keras
    ├── robust_scaler.pkl
    ├── feature_list.json
    └── inference.py                  # Deployment inference script
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place creditcard.csv in this directory

# 3. Run the pipeline
python credit_card_fraud_detection.py

# 4. Check output/ for all artifacts, visualizations, and trained models
```

## Key Features

- **Feature Engineering:** Temporal patterns, amount deviations, PCA magnitude, interaction features
- **Evaluation on Imbalanced Test Set:** Realistic deployment conditions (no balanced test set inflation)
- **Threshold Optimization:** Data-driven threshold selection balancing precision vs recall
- **Business Impact Analysis:** Dollar-value quantification of fraud prevented vs investigation costs
- **Automated Deployment:** Inference script with risk scoring (LOW/MEDIUM/HIGH)

## Evaluation Metrics

- Precision, Recall, F1-Score (primary)
- AUC-ROC, AUC-PR
- Matthews Correlation Coefficient (MCC)
- Confusion matrices on original imbalanced test set

## References

1. Popova & Gardi (2024) — Hybrid sampling strategies for fraud detection
2. Fariha et al. (2025) — Behavioral feature engineering for fraud detection
3. Marazqah Btoush et al. (2023) — Systematic review of ML/DL fraud detection
4. Mienye & Jere (2024) — Deep learning approaches for fraud detection
5. Alarfaj et al. (2022) — CNN architectures for fraud detection
6. Karunya et al. (2025) — Probability-based kNN for imbalanced fraud data

## Tools & Hardware

- **Framework:** Python 3.10+, scikit-learn, XGBoost, TensorFlow/Keras
- **Environment:** Anaconda / Jupyter / Google Colab
- **Hardware:** CPU (sufficient), GPU recommended for DL models

## Vistualization Dashboard(notebook)
[View the txt File](./analysis_notebook_report.txt)

