# Log Anomaly Detector

AI-powered system log analyzer using unsupervised machine learning (Isolation Forest).

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn main:app --reload
```

### 3. Train on your logs
```bash
curl -X POST http://localhost:8000/train \
  -F "file=@logs/sample/auth.log"
```

### 4. Analyze new logs
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@logs/sample/auth.log"
```

### 5. Docker (optional)
```bash
docker build -t log-anomaly-detector .
docker run -p 8000:8000 log-anomaly-detector
```

## Project structure
```
log-anomaly-detector/
├── main.py          # FastAPI app (routes)
├── parser.py        # Log line parsing (regex)
├── features.py      # Feature extraction
├── detector.py      # Isolation Forest model
├── requirements.txt
├── Dockerfile
├── models/          # Saved model (auto-created on train)
└── logs/sample/     # Sample logs for testing
```

## Week-by-week plan

**Week 1** — Get parsing and features working on real data
- Download LANL dataset: https://csr.lanl.gov/data/cyber1/
- Extend parser.py with more regex patterns
- Run `python -c "from parser import *; print(lines_to_df(open('logs/sample/auth.log').readlines()))"` to verify

**Week 2** — Train and evaluate the model
- Split your dataset into train/test
- Evaluate with: precision, recall on known attack events
- Try different `contamination` values (0.01–0.10)

**Week 3** — Wrap and ship
- Add a simple HTML dashboard (fetch /analyze, display results)
- Dockerize and push to Docker Hub or fly.io
- Write a 1-page project summary for your portfolio
