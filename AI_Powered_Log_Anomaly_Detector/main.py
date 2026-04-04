"""
main.py — FastAPI application exposing two endpoints:

  POST /train   — Upload log lines to train the model
  POST /analyze — Upload log lines to get anomaly predictions
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from parser   import lines_to_df
from features import extract_features
from detector import LogAnomalyDetector

app = FastAPI(title="Log Anomaly Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = LogAnomalyDetector(contamination=0.05)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _process_upload(file_bytes: bytes):
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    df    = lines_to_df(lines)
    if df.empty:
        raise HTTPException(status_code=422, detail="No parseable log lines found.")
    feats = extract_features(df)
    return df, feats

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/train")
async def train(file: UploadFile = File(...)):
    """Train the model on uploaded log file."""
    raw = await file.read()
    df, feats = _process_upload(raw)
    detector.train(feats)
    return {"status": "trained", "samples": len(feats)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded log file and return flagged anomalies."""
    raw = await file.read()
    df, feats = _process_upload(raw)

    if not detector.trained:
        try:
            detector.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="No trained model found. POST to /train first."
            )

    results = detector.predict(feats)
    df["anomaly_score"] = results["anomaly_score"].round(4)
    df["is_anomaly"]    = results["is_anomaly"]

    anomalies = df[df["is_anomaly"] == True][["raw", "anomaly_score"]].to_dict("records")
    return JSONResponse({
        "total_lines"   : len(df),
        "anomalies_found": len(anomalies),
        "anomalies"     : anomalies[:50],  # cap response size
    })

@app.get("/health")
def health():
    return {"status": "ok", "model_trained": detector.trained}
