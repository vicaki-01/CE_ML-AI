"""
parser.py — Ingest and parse raw log lines into structured feature rows.
Supports: auth.log (SSH/PAM), Apache/Nginx access logs, generic syslog.
"""
import re
import pandas as pd
from datetime import datetime

# ── Regex patterns ────────────────────────────────────────────────────────────
PATTERNS = {
    "auth": re.compile(
        r"(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<time>[\d:]+)\s+(?P<host>\S+)\s+"
        r"(?P<service>\S+?)(\[(?P<pid>\d+)\])?:\s+(?P<message>.+)"
    ),
    "apache": re.compile(
        r"(?P<ip>[\d\.]+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+"
        r"\"(?P<method>\w+)\s+(?P<path>\S+)\s+\S+\"\s+(?P<status>\d+)\s+(?P<size>\d+)"
    ),
}

def parse_line(line: str) -> dict | None:
    """Return a flat dict of fields, or None if no pattern matches."""
    for log_type, pat in PATTERNS.items():
        m = pat.match(line.strip())
        if m:
            d = m.groupdict()
            d["log_type"] = log_type
            d["raw"] = line.strip()
            return d
    return None

def lines_to_df(lines: list[str]) -> pd.DataFrame:
    rows = [parse_line(l) for l in lines if parse_line(l)]
    return pd.DataFrame(rows) if rows else pd.DataFrame()
