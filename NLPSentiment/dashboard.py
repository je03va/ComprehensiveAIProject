"""
dashboard.py: a simple Streamlit dashboard over the saved results.
Run:  streamlit run dashboard.py      (after python sentiment_agent.py)
It only reads files from outputs/, so it is fast and works offline.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sentiment_agent import (LABEL_COLORS, LABELS, MODELS, NEGATIVE_CUTOFF, OUT_DIR,
                             POSITIVE_CUTOFF)

st.set_page_config(page_title="Developer Sentiment Analyzer", layout="wide")
st.title("Can off-the-shelf sentiment models understand developers?")

summary_path = os.path.join(OUT_DIR, "results_summary.csv")
if not os.path.exists(summary_path):
    st.error("No results yet. Run `python sentiment_agent.py` first.")
    st.stop()

summary = pd.read_csv(summary_path)
datasets = summary["dataset"].unique().tolist()

# --- sidebar: pick dataset and model ---
dataset = st.sidebar.selectbox("Dataset", datasets)
model = st.sidebar.selectbox("Model", MODELS, index=MODELS.index("roberta"))
color_by = st.sidebar.radio("Color scatter by", ["true label", f"predicted label ({model})"])
preds = pd.read_csv(os.path.join(OUT_DIR, f"predictions_{dataset}.csv"), keep_default_na=False)

# --- 1. results table (Innovation #1) ---
st.header("1. Model comparison")
row = summary[(summary.dataset == dataset) & (summary.model == model)].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{row.accuracy:.2f}")
c2.metric("Macro F1", f"{row.macro_f1:.2f}")
c3.metric("Neutral posts called negative", f"{row.neutral_called_negative:.0%}")
c4.metric("Runtime", f"{row.runtime_sec:.1f}s")
st.dataframe(summary, width="stretch", hide_index=True)
st.image(os.path.join(OUT_DIR, "model_comparison.png"))

# --- 2. confusion matrix + scatter (extends the professor's plot) ---
st.header(f"2. {model} on {dataset}")
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(preds["label"], preds[f"pred_{model}"], labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
with right:
    st.subheader("Polarity vs subjectivity (TextBlob)")
    column = "label" if color_by == "true label" else f"pred_{model}"
    fig, ax = plt.subplots(figsize=(5, 4))
    for label in LABELS:
        part = preds[preds[column] == label]
        ax.scatter(part["polarity"], part["subjectivity"], s=12, alpha=0.6,
                   color=LABEL_COLORS[label], label=label, edgecolors="white", linewidths=0.3)
    ax.axvline(POSITIVE_CUTOFF, color="#c3c2b7", lw=1, ls="--")
    ax.axvline(NEGATIVE_CUTOFF, color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("polarity")
    ax.set_ylabel("subjectivity")
    ax.legend(frameon=False, title=color_by)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

st.subheader("Browse posts the model got wrong")
wrong = preds[preds["label"] != preds[f"pred_{model}"]]
st.caption(f"{len(wrong)} of {len(preds)} posts misclassified")
st.dataframe(wrong[["text", "label", f"pred_{model}", f"score_{model}"]], width="stretch", hide_index=True)

# --- 3. agreement (Innovation #2) ---
st.header("3. Do the models agree?")
st.dataframe(pd.read_csv(os.path.join(OUT_DIR, "agreement_summary.csv")), width="stretch", hide_index=True)
st.caption("Posts where TextBlob, VADER and RoBERTa disagree:")
st.dataframe(pd.read_csv(os.path.join(OUT_DIR, f"disagreements_{dataset}.csv")), width="stretch", hide_index=True)

# --- 4. tuning (Innovation #3) ---
st.header("4. Tech-lexicon tuning: VADER before vs after")
st.dataframe(pd.read_csv(os.path.join(OUT_DIR, "tuning_examples.csv")), width="stretch", hide_index=True)
