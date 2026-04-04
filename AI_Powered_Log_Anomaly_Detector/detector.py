"""
detector.py — Train and run the Isolation Forest anomaly detector.

Isolation Forest works by randomly partitioning data. Anomalies are
isolated in fewer splits — they get a lower (more negative) score.
Threshold: score < -0.05 → flag as anomaly.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle, pathlib

MODEL_PATH = pathlib.Path("models/isolation_forest.pkl")

class LogAnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        """
        contamination: expected fraction of anomalies in training data.
        0.05 = assume ~5% of your training logs are unusual.
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        self.trained = False

    def train(self, feature_df: pd.DataFrame):
        """Fit the model on a clean(ish) feature matrix."""
        self.model.fit(feature_df.values)
        self.trained = True
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model trained on {len(feature_df)} samples and saved.")

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        self.trained = True

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the feature_df with two new columns:
          anomaly_score  — raw IF score (lower = more anomalous)
          is_anomaly     — True if score < -0.05
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call .train() first.")
        scores = self.model.score_samples(feature_df.values)
        results = feature_df.copy()
        results["anomaly_score"] = scores
        results["is_anomaly"]    = scores < -0.05
        return results
