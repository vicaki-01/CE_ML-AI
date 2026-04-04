"""
features.py — Turn parsed log DataFrames into numeric feature matrices.

Features extracted per log line:
  - hour_of_day       (0-23) — unusual login hours are a key signal
  - is_failed_auth    (0/1)  — "Failed password" or "Invalid user"
  - is_root_attempt   (0/1)  — attempts targeting root account
  - ip_freq           (int)  — how many times this source IP appears in window
  - service_encoded   (int)  — label-encoded service name (sshd, sudo, etc.)
  - message_len       (int)  — longer messages sometimes indicate payloads
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

_le_service = LabelEncoder()

FAILED_KEYWORDS = ["failed password", "invalid user", "authentication failure"]
ROOT_KEYWORDS   = ["root", "su:", "sudo"]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame()

    # Hour of day (auth logs carry HH:MM:SS)
    def _hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return 12  # default to noon if unparseable
    feats["hour_of_day"] = df.get("time", pd.Series(dtype=str)).apply(_hour)

    # Auth-specific signals
    msg = df.get("message", pd.Series([""] * len(df))).fillna("").str.lower()
    feats["is_failed_auth"]  = msg.apply(lambda m: int(any(k in m for k in FAILED_KEYWORDS)))
    feats["is_root_attempt"] = msg.apply(lambda m: int(any(k in m for k in ROOT_KEYWORDS)))

    # IP frequency within this batch (proxy for brute-force)
    ip_col = df.get("ip", pd.Series(["0.0.0.0"] * len(df))).fillna("0.0.0.0")
    ip_counts = ip_col.value_counts()
    feats["ip_freq"] = ip_col.map(ip_counts).fillna(1).astype(int)

    # Service label encoding
    svc_col = df.get("service", pd.Series(["unknown"] * len(df))).fillna("unknown")
    try:
        feats["service_encoded"] = _le_service.fit_transform(svc_col)
    except Exception:
        feats["service_encoded"] = 0

    # Message length
    feats["message_len"] = msg.str.len().fillna(0).astype(int)

    return feats.fillna(0)
