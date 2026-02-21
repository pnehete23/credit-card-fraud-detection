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

## Visualization Dashboard(notebook)
[Original file](./analysis_notebook_report.txt)
======================================================================
CREDIT CARD FRAUD DETECTION - MODEL PERFORMANCE ANALYSIS
======================================================================
Analysis Date: 2026-02-20 23:33:25
Dataset: Kaggle Credit Card Fraud Dataset
======================================================================

1. DATA OVERVIEW
----------------------------------------------------------------------
              Model  Precision   Recall  F1-Score  AUC-ROC   AUC-PR      MCC  Train Time (s)
            XGBoost   0.873563 0.800000  0.835165 0.971256 0.809437 0.835711        8.817314
      Random Forest   0.852273 0.789474  0.819672 0.962272 0.804110 0.819983      141.552637
     Neural Network   0.625000 0.789474  0.697674 0.939289 0.762756 0.701887       41.821549
        Autoencoder   0.607595 0.505263  0.551724 0.952289 0.458935 0.553393       16.235340
Logistic Regression   0.043295 0.863158  0.082453 0.957465 0.731708 0.189178       47.237501

2. STATISTICAL SUMMARY
----------------------------------------------------------------------
       Precision    Recall  F1-Score   AUC-ROC    AUC-PR       MCC  Train Time (s)
count   5.000000  5.000000  5.000000  5.000000  5.000000  5.000000        5.000000
mean    0.600345  0.749474  0.597338  0.956514  0.713389  0.620031       51.132868
std     0.335068  0.139925  0.309535  0.011894  0.145753  0.266118       53.117918
min     0.043295  0.505263  0.082453  0.939289  0.458935  0.189178        8.817314
25%     0.607595  0.789474  0.551724  0.952289  0.731708  0.553393       16.235340
50%     0.625000  0.789474  0.697674  0.957465  0.762756  0.701887       41.821549
75%     0.852273  0.800000  0.819672  0.962272  0.804110  0.819983       47.237501
max     0.873563  0.863158  0.835165  0.971256  0.809437  0.835711      141.552637

3. MODEL RANKINGS BY METRIC
----------------------------------------------------------------------

Precision:
  1. XGBoost              0.8736
  2. Random Forest        0.8523
  3. Neural Network       0.6250
  4. Autoencoder          0.6076
  5. Logistic Regression  0.0433

Recall:
  5. Logistic Regression  0.8632
  1. XGBoost              0.8000
  2. Random Forest        0.7895
  3. Neural Network       0.7895
  4. Autoencoder          0.5053

F1-Score:
  1. XGBoost              0.8352
  2. Random Forest        0.8197
  3. Neural Network       0.6977
  4. Autoencoder          0.5517
  5. Logistic Regression  0.0825

AUC-ROC:
  1. XGBoost              0.9713
  2. Random Forest        0.9623
  5. Logistic Regression  0.9575
  4. Autoencoder          0.9523
  3. Neural Network       0.9393

AUC-PR:
  1. XGBoost              0.8094
  2. Random Forest        0.8041
  3. Neural Network       0.7628
  5. Logistic Regression  0.7317
  4. Autoencoder          0.4589

MCC:
  1. XGBoost              0.8357
  2. Random Forest        0.8200
  3. Neural Network       0.7019
  4. Autoencoder          0.5534
  5. Logistic Regression  0.1892

Training Time (Fastest to Slowest):
  1. XGBoost              8.82s
  4. Autoencoder          16.24s
  3. Neural Network       41.82s
  5. Logistic Regression  47.24s
  2. Random Forest        141.55s

4. BEST MODEL IDENTIFICATION
----------------------------------------------------------------------
Best Overall Model: XGBoost
Composite Score: 0.8680

Performance Breakdown:
  Precision:     0.8736
  Recall:        0.8000
  F1-Score:      0.8352
  AUC-ROC:       0.9713
  AUC-PR:        0.8094
  MCC:           0.8357
  Training Time: 8.82s

5. EFFICIENCY ANALYSIS
----------------------------------------------------------------------
              Model  Train Time (s)  F1-Score  Efficiency_Ratio
            XGBoost        8.817314  0.835165          5.683124
        Autoencoder       16.235340  0.551724          2.038975
     Neural Network       41.821549  0.697674          1.000931
      Random Forest      141.552637  0.819672          0.347435
Logistic Regression       47.237501  0.082453          0.104731

Note: Efficiency Ratio = F1-Score / Training Time (minutes)
Higher values indicate better performance relative to training time

6. PRECISION-RECALL TRADE-OFF ANALYSIS
----------------------------------------------------------------------
              Model  Precision   Recall  F1-Score  PR_Balance
            XGBoost   0.873563 0.800000  0.835165    0.835165
      Random Forest   0.852273 0.789474  0.819672    0.819672
     Neural Network   0.625000 0.789474  0.697674    0.697674
        Autoencoder   0.607595 0.505263  0.551724    0.551724
Logistic Regression   0.043295 0.863158  0.082453    0.082453

Analysis:
  ✓ XGBoost: Excellent balance (High Precision & High Recall)
  ✓ Random Forest: Excellent balance (High Precision & High Recall)
  ⚠ Autoencoder: Moderate performance - room for improvement
  ✗ Logistic Regression: Poor precision - too many false positives

7. AUC METRICS COMPARISON
----------------------------------------------------------------------
              Model  AUC-ROC   AUC-PR  AUC_Gap
            XGBoost 0.971256 0.809437 0.161819
      Random Forest 0.962272 0.804110 0.158161
Logistic Regression 0.957465 0.731708 0.225757
        Autoencoder 0.952289 0.458935 0.493354
     Neural Network 0.939289 0.762756 0.176534

Interpretation:
  - AUC-ROC: Overall model discrimination ability
  - AUC-PR: Performance on imbalanced data (more relevant for fraud)
  - Gap: Larger gaps suggest challenges with class imbalance

8. PERFORMANCE VS TRAINING TIME ANALYSIS
----------------------------------------------------------------------
              Model  F1-Score  Train Time (s)
            XGBoost  0.835165        8.817314
      Random Forest  0.819672      141.552637
     Neural Network  0.697674       41.821549
        Autoencoder  0.551724       16.235340
Logistic Regression  0.082453       47.237501

Fastest: XGBoost (8.82s)
Slowest: Random Forest (141.55s)
Speed Difference: 16.1x

9. KEY FINDINGS & RECOMMENDATIONS
======================================================================

1. Best Overall Model:
   XGBoost with F1-Score of 0.8352 and AUC-ROC of 0.9713

2. Fastest Training:
   XGBoost completes training in just 8.82 seconds

3. Best Recall:
   Logistic Regression achieves highest recall (0.8632), but with very low precision (0.0433)

4. Most Balanced:
   XGBoost shows best precision-recall balance

5. Production Recommendation:
   XGBoost is recommended for production deployment due to:
     - Highest overall performance metrics
     - Fast training time (8.82s)
     - Excellent balance between precision and recall
     - Best AUC-ROC score (0.9713)

10. IMPLEMENTATION CONSIDERATIONS
======================================================================

1. Class Imbalance: All models handle the highly imbalanced fraud dataset differently. Monitor AUC-PR closely as it's more informative than AUC-ROC for imbalanced data.

2. False Positive Cost: In fraud detection, false positives (legitimate transactions flagged) can frustrate customers. XGBoost and Random Forest offer the best precision (>0.85).

3. False Negative Cost: Missing actual fraud is costly. While Logistic Regression has highest recall (0.86), its low precision (0.04) makes it impractical - it would flag 96% incorrectly.

4. Scalability: XGBoost's fast training time (8.8s) vs Random Forest (141.6s) is crucial for frequent model retraining as fraud patterns evolve.

5. Interpretability: While Neural Networks and Autoencoders offer deep learning capabilities, tree-based models (XGBoost, Random Forest) provide better interpretability for regulatory compliance.

6. Real-time Scoring: All models can perform real-time inference, but XGBoost typically offers the best latency-performance trade-off for production systems.

7. Recommended Threshold Tuning: Start with XGBoost and adjust the classification threshold based on the specific cost ratio of false positives to false negatives in your use case.

======================================================================
END OF ANALYSIS
======================================================================

For interactive visualizations, please open the HTML dashboard.

SUMMARY STATISTICS
----------------------------------------------------------------------
Best Model: XGBoost
Best F1-Score: 0.8351648351648352
Best AUC-ROC: 0.9712557682356144
Fastest Training: XGBoost
Training Time Range: 8.82s - 141.55s
Analysis Date: 2026-02-20 23:33:25

