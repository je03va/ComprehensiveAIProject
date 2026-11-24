"""
prepare_data.py - Download the labeled developer-sentiment datasets and convert
each one to a simple CSV with two columns: text, label
(label is one of: negative / neutral / positive).

Run once:  python prepare_data.py

"""
import os
import requests
import pandas as pd

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

SENTI4SD_BASE = ("https://raw.githubusercontent.com/collab-uniba/Senti4SD/master/"
                 "Senti4SD_GoldStandard_and_DSM/Senti4SD_Train_Test_Partitions/")
SOURCES = {
    "train3098itemPOLARITY.csv": SENTI4SD_BASE + "train3098itemPOLARITY.csv",
    "test1326itemPOLARITY.csv": SENTI4SD_BASE + "test1326itemPOLARITY.csv",
    "github_gold.csv": "https://ndownloader.figshare.com/files/21001260",
}
VALID_LABELS = {"negative", "neutral", "positive"}


def download_raw_files():
    """Download each raw file once (skip if it's already in data/raw)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            print(f"  already have {path}")
            continue
        print(f"  downloading {name} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()  # stop loudly if a download is blocked
        with open(path, "wb") as f:
            f.write(resp.content)


def clean_labels(df):
    """Keep only text + label, normalize the label, drop empty/odd rows."""
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # \r\n -> \n so the text round-trips through CSV exactly (the RoBERTa cache matches on text)
    df["text"] = df["text"].astype(str).str.replace("\r\n", "\n").str.strip()
    df = df[df["label"].isin(VALID_LABELS) & (df["text"] != "")]
    return df.reset_index(drop=True)


def prepare_senti4sd():
    """Senti4SD files have no header: id;label;text (semicolon-separated).
    We merge train + test because we only *evaluate* pre-trained models (no training)."""
    parts = []
    for name in ["train3098itemPOLARITY.csv", "test1326itemPOLARITY.csv"]:
        part = pd.read_csv(os.path.join(RAW_DIR, name), sep=";", header=None,
                           names=["id", "label", "text"], encoding="utf-8")
        parts.append(part)
    return clean_labels(pd.concat(parts, ignore_index=True))


def prepare_github():
    """github_gold.csv has a header: ID;Polarity;Text (semicolon-separated).
    Many texts end with a stray '\"' left over from the original export - strip it."""
    df = pd.read_csv(os.path.join(RAW_DIR, "github_gold.csv"), sep=";", encoding="utf-8")
    df = df.rename(columns={"Polarity": "label", "Text": "text"})
    df["text"] = df["text"].astype(str).str.rstrip('"').str.strip()
    return clean_labels(df)


def summarize(name, df):
    """Print row count and label balance to sanity-check the data."""
    counts = df["label"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print(f"\n{name}: {len(df)} rows")
    for label in ["negative", "neutral", "positive"]:
        print(f"  {label:<9} {counts.get(label, 0):>5}  ({pct.get(label, 0)}%)")


if __name__ == "__main__":
    print("Downloading raw files...")
    download_raw_files()

    datasets = {"stackoverflow": prepare_senti4sd(), "github": prepare_github()}
    for name, df in datasets.items():
        out_path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)
        summarize(name, df)
        print(f"  saved -> {out_path}")
