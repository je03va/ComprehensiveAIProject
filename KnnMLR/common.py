"""Shared setup: reuses the kNN assignment's data loading, percept cleaning and kNN pipeline."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
KNN_DIR = BASE_DIR.parent / "kNNRegressorAgent"
sys.path.insert(0, str(KNN_DIR))
import agent as knn_base  # noqa: E402  (the existing kNN assignment, imported, never modified)

from agent import CATEGORICAL, NUMERIC, RANDOM_STATE, TARGET  # noqa: E402,F401

DATA_PATH = BASE_DIR / "data" / "transcoding_mesurment.tsv"
if not DATA_PATH.exists():  # fall back to wherever the kNN assignment reads it from
    DATA_PATH = Path(knn_base.DATA_PATH)
LOG_DIR = BASE_DIR / "logs"

TEST_SIZE = 0.2
KNN_CONFIG = {"features": "all", "k": 3, "weights": "distance"}  # tuned in the kNN assignment
CODEC_ALIASES = {"h.264": "h264", "avc": "h264", "x264": "h264", "mpeg-4": "mpeg4", "mp4v": "mpeg4",
                 "xvid": "mpeg4", "vp-8": "vp8", "flash": "flv", "sorenson": "flv"}


def load_data(path=DATA_PATH):
    """Load the TSV and return X, y. id and umem are dropped (umem is only known after the job runs)."""
    df = pd.read_csv(path, sep="\t").drop(columns=knn_base.LEAKAGE)
    return df.drop(columns=TARGET), df[TARGET]


def build_knn():
    """The tuned kNN from the kNN assignment, built with its own pipeline function."""
    return knn_base.build_pipeline(knn_base.FEATURE_SETS[KNN_CONFIG["features"]],
                                   KNN_CONFIG["k"], KNN_CONFIG["weights"])


def clean_percept_v2(raw):
    """Extends the kNN clean_percept with codec aliases (e.g. 'H.264' -> 'h264') and a frame-count check.

    Returns (job, warnings).
    """
    raw = dict(raw)
    notes = []
    for col in CATEGORICAL:
        value = str(raw.get(col) or "").strip().lower()
        if value in CODEC_ALIASES:
            notes.append(f"{col}: '{raw[col]}' read as '{CODEC_ALIASES[value]}'.")
            raw[col] = CODEC_ALIASES[value]
    job, warnings = knn_base.clean_percept(raw)

    i, p, b, frames = (job[c] for c in ("i", "p", "b", "frames"))
    if not any(pd.isna(v) for v in (i, p, b, frames)) and frames != i + p + b:
        notes.append(f"frames={frames:g} but i+p+b={i + p + b:g}; using i+p+b.")
        job["frames"] = i + p + b
    return job, warnings + notes


def log_csv(filename, row):
    """Append one timestamped row to a CSV in this project's logs/ folder."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / filename
    entry = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **row}])
    entry.to_csv(path, mode="a", header=not path.exists(), index=False)
