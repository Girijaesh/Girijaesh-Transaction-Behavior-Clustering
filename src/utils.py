from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ccdata.csv"
REPORTS_DIR = BASE_DIR / "reports"
CUSTOMER_SEGMENTS_PATH = REPORTS_DIR / "customer_segments.csv"
CLUSTER_SUMMARY_PATH = REPORTS_DIR / "cluster_summary.csv"


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR
