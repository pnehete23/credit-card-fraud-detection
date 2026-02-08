"""
Inference Script — Credit Card Fraud Detection
Usage: python inference.py <transaction_json>
"""
import pickle
import json
import numpy as np
import sys

# Load artifacts
with open("output/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("output/robust_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("output/feature_list.json", "r") as f:
    features = json.load(f)

def predict_fraud(transaction: dict, threshold: float = 0.45):
    """Predict if a transaction is fraudulent.

    Args:
        transaction: dict with feature values
        threshold: classification threshold (default optimized)

    Returns:
        dict with prediction and probability
    """
    X = np.array([[transaction.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    return {
        "is_fraud": bool(prob >= threshold),
        "fraud_probability": float(prob),
        "risk_level": "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.5 else "LOW",
        "threshold_used": threshold
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        txn = json.loads(sys.argv[1])
        result = predict_fraud(txn)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python inference.py '<json_transaction>'")
