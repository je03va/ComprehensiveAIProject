"""
sentiment_agent.py - Developer Sentiment Comparative Analyzer

Research question: can off-the-shelf sentiment models understand how software
developers talk? ("kill the process", "fatal error", "this bug is nasty" ...)

Builds on the professor's SentimentTextblob class (same method names and flow:
load_data -> compute_sentiment -> quantitative_sentiment -> visualize_sentiment ->
evaluation_confusion_matrix), extended from 1 model / 2 classes to
4 models / 3 classes (negative, neutral, positive):
    textblob     - TextBlob polarity with a neutral band
    vader        - VADER compound score, standard +/-0.05 thresholds
    vader_tuned  - VADER + tech-domain lexicon + text cleanup (Innovation #3)
    roberta      - Hugging Face cardiffnlp/twitter-roberta-base-sentiment-latest

Run:  python sentiment_agent.py      (after python prepare_data.py)
"""
import logging
import os
import re
import time

import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ----------------------------- settings ---------------------------------
DATASETS = {"stackoverflow": "data/stackoverflow.csv", "github": "data/github.csv"}
SAMPLE_SIZE = 1000          # rows per dataset; set to None to use every row
RANDOM_SEED = 42
OUT_DIR = "outputs"

LABELS = ["negative", "neutral", "positive"]
MODELS = ["textblob", "vader", "vader_tuned", "roberta"]
BASE_MODELS = ["textblob", "vader", "roberta"]  # the untouched models

# Professor's binary version used POSITIVE_CUTOFF = 0. With 3 classes we need a
# neutral band, because most dev posts have a polarity near 0.
POSITIVE_CUTOFF = 0.1
NEGATIVE_CUTOFF = -0.1
VADER_CUTOFF = 0.05         # standard VADER thresholds from the VADER paper

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_CACHE = "data/roberta_cache.csv"   # transformer runs once, then is read from here
BATCH_SIZE = 32

# Colors: red / gray / blue for negative / neutral / positive (gray = "no sentiment").
LABEL_COLORS = {"negative": "#e34948", "neutral": "#8a8984", "positive": "#2a78d6"}
MODEL_COLORS = {"textblob": "#2a78d6", "vader": "#eb6834", "vader_tuned": "#1baf7a", "roberta": "#eda100"}

"""Tech-domain lexicon. VADER was built for social media, so it reads
technical words as emotional. Values use VADER's -4..+4 scale.
("bug", "deprecated", "exception", "abort" are already neutral in VADER, so not listed.)
"""
TECH_LEXICON = {
    # "scary" words with a technical meaning -> neutral (0)
    "kill": 0.0, "killed": 0.0, "killing": 0.0, "kills": 0.0,
    "fatal": 0.0, "exploit": 0.0, "exploits": 0.0, "attack": 0.0, "attacks": 0.0,
    "dead": 0.0, "deadlock": 0.0, "crash": 0.0, "error": 0.0, "errors": 0.0,
    "warning": 0.0, "block": 0.0, "blocked": 0.0, "reject": 0.0, "rejected": 0.0,
    "destroy": 0.0, "destroyed": 0.0, "panic": 0.0, "dump": 0.0, "violation": 0.0,
    "threat": 0.0, "vulnerable": 0.0, "vulnerability": 0.0, "risk": 0.0,
    "critical": 0.0, "severe": 0.0, "defect": 0.0, "forced": 0.0, "lazy": 0.0,
    "argument": 0.0, "arguments": 0.0, "empty": 0.0, "missing": 0.0,
    # "happy" words with a technical meaning -> neutral (0). Found by checking which
    # words pushed truly-neutral posts to "positive" ("git commit", "return true", "free memory").
    "commit": 0.0, "committed": 0.0, "value": 0.0, "values": 0.0, "true": 0.0,
    "solution": 0.0, "solutions": 0.0, "create": 0.0, "created": 0.0,
    "security": 0.0, "secure": 0.0, "support": 0.0, "supported": 0.0,
    "free": 0.0, "freed": 0.0, "clean": 0.0, "original": 0.0,
    # everyday dev words that are only mildly negative ("the test failed")
    "fail": -0.5, "failed": -0.5, "fails": -0.5, "failing": -0.5, "failure": -0.5,
    "problem": -0.5, "problems": -0.5,
    # developer slang VADER doesn't know
    "lgtm": 2.0, "hacky": -1.0, "kludge": -1.0, "wontfix": -0.5,
}

LOG = logging.getLogger("sentiment_agent")


# ----------------------------- helpers ----------------------------------
def setup_logging():
    """exports + logging. Log to the console and to outputs/run.log
    (every step also saves its tables as CSV and charts as PNG in outputs/)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(OUT_DIR, "run.log"), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(file_handler)
    LOG.addHandler(logging.StreamHandler())


def clean_dev_text(text):
    """light cleanup. Code and URLs carry no emotion but
    often contain words like 'kill' or 'error' that confuse lexicon models."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)   # fenced code blocks
    text = re.sub(r"`[^`]*`", " ", text)                      # inline code
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)        # URLs
    return re.sub(r"\s+", " ", text).strip()


def to_label(score, pos_cutoff, neg_cutoff):
    """Turn a numeric score into negative / neutral / positive."""
    if score > pos_cutoff:
        return "positive"
    if score < neg_cutoff:
        return "negative"
    return "neutral"


def make_tuned_vader():
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(TECH_LEXICON)  # add the tech-domain lexicon to the standard VADER lexicon
    return analyzer


VADER = SentimentIntensityAnalyzer()
VADER_TUNED = make_tuned_vader()
_roberta_pipeline = None  # loaded only when needed (it is slow to start)


def score_roberta(texts):
    """Score texts with the transformer, using a CSV cache so each text is only run once.
    Returns (labels, scores, seconds) where score = P(positive) - P(negative)."""
    global _roberta_pipeline
    cache = {}
    if os.path.exists(ROBERTA_CACHE):
        cached = pd.read_csv(ROBERTA_CACHE, keep_default_na=False)
        cache = {r.text: (r.label, r.score, r.seconds) for r in cached.itertuples()}

    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        if _roberta_pipeline is None:
            from transformers import pipeline
            _roberta_pipeline = pipeline("sentiment-analysis", model=ROBERTA_MODEL,
                                         top_k=None, device=-1)  # device=-1 = CPU
        LOG.info(f"  RoBERTa: scoring {len(missing)} new texts on CPU (already cached: {len(set(texts)) - len(missing)})")
        for start in range(0, len(missing), 256):  # chunks, so we can show progress
            chunk = missing[start:start + 256]
            t0 = time.time()
            results = _roberta_pipeline(chunk, batch_size=BATCH_SIZE, truncation=True, max_length=512)
            sec_per_text = (time.time() - t0) / len(chunk)
            for text, result in zip(chunk, results):
                probs = {d["label"].lower(): d["score"] for d in result}
                cache[text] = (max(probs, key=probs.get), probs["positive"] - probs["negative"], sec_per_text)
            LOG.info(f"    {min(start + 256, len(missing))}/{len(missing)} done")
        pd.DataFrame([(t, *v) for t, v in cache.items()],
                     columns=["text", "label", "score", "seconds"]).to_csv(ROBERTA_CACHE, index=False)

    labels = [cache[t][0] for t in texts]
    scores = [cache[t][1] for t in texts]
    seconds = sum(cache[t][2] for t in texts)  # measured time, even if read from cache
    return labels, scores, seconds


def predict(texts, model):
    """Score a list of texts with one model. Returns (labels, scores)."""
    if model == "textblob":
        scores = [TextBlob(t).sentiment.polarity for t in texts]
        return [to_label(s, POSITIVE_CUTOFF, NEGATIVE_CUTOFF) for s in scores], scores
    if model in ("vader", "vader_tuned"):
        analyzer = VADER_TUNED if model == "vader_tuned" else VADER
        prepared = [clean_dev_text(t) for t in texts] if model == "vader_tuned" else texts
        scores = [analyzer.polarity_scores(t)["compound"] for t in prepared]
        return [to_label(s, VADER_CUTOFF, -VADER_CUTOFF) for s in scores], scores
    labels, scores, _ = score_roberta(texts)
    return labels, scores


# ------------------- main class  --------------------
class SentimentAgent:
    """One labeled dataset, scored and evaluated by every model."""

    def __init__(self, name):
        self.name = name
        self.df = None
        self.runtime = {}

    def load_data(self, file_path):
        """Load a CSV with columns text + label, optionally sample it down to SAMPLE_SIZE rows."""
        self.df = pd.read_csv(file_path, keep_default_na=False)
        if SAMPLE_SIZE and len(self.df) > SAMPLE_SIZE:
            self.df = self.df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        LOG.info(f"\n=== {self.name}: loaded {len(self.df)} rows from {file_path}")

    def compute_sentiment(self):
        """Score every row with every model and time each one."""
        texts = self.df["text"].tolist()
        # compute TextBlob polarity and subjectivity (for the scatter plot)
        blobs = [TextBlob(t).sentiment for t in texts]
        self.df["polarity"] = [b.polarity for b in blobs]
        self.df["subjectivity"] = [b.subjectivity for b in blobs]

        for model in MODELS:  # several models on the same data
            t0 = time.time()
            labels, scores = predict(texts, model)
            self.runtime[model] = time.time() - t0
            self.df[f"pred_{model}"] = labels
            self.df[f"score_{model}"] = scores
        # For RoBERTa use the measured model time (stored in the cache), not the cache-read time.
        self.runtime["roberta"] = score_roberta(texts)[2]

    def quantitative_sentiment(self):
        """Print a table of the first few rows, and a summary of the numeric columns."""
        LOG.info(self.df[["text", "label", "pred_vader", "pred_roberta"]].head().to_string(max_colwidth=50))
        LOG.info(self.df[["polarity", "subjectivity", "score_vader", "score_vader_tuned",
                          "score_roberta"]].describe().round(3).to_string())

    def visualize_sentiment(self):
        """Scatter plot: polarity vs subjectivity, colored by true label."""
        fig, ax = plt.subplots(figsize=(7, 5))
        for label in LABELS:
            part = self.df[self.df["label"] == label]
            ax.scatter(part["polarity"], part["subjectivity"], s=16, alpha=0.6,
                       color=LABEL_COLORS[label], label=f"{label} (true)", edgecolors="white", linewidths=0.4)
        ax.axvline(POSITIVE_CUTOFF, color="#c3c2b7", lw=1, ls="--")
        ax.axvline(NEGATIVE_CUTOFF, color="#c3c2b7", lw=1, ls="--")
        ax.set_xlabel("TextBlob polarity  (dashed lines = neutral band)")
        ax.set_ylabel("TextBlob subjectivity")
        ax.set_title(f"{self.name}: polarity vs subjectivity by true label")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"scatter_{self.name}.png"), dpi=120)
        plt.close(fig)

    def evaluation_confusion_matrix(self):
        """Applies accuracy, macro F1 and a confusion matrix for every model."""
        rows = []
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4))
        for ax, model in zip(axes, MODELS):
            y_true, y_pred = self.df["label"], self.df[f"pred_{model}"]
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro")
            LOG.info(f"\n--- {self.name} / {model}: accuracy={acc:.3f}  macro F1={f1:.3f}")
            LOG.info(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=LABELS)
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{model}\nacc {acc:.2f} | F1 {f1:.2f}")
            # how often a truly neutral post is called negative = the "dev jargon" failure
            neutral_as_neg = ((y_true == "neutral") & (y_pred == "negative")).sum() / max((y_true == "neutral").sum(), 1)
            rows.append({"dataset": self.name, "model": model, "accuracy": round(acc, 3),
                         "macro_f1": round(f1, 3), "neutral_called_negative": round(neutral_as_neg, 3),
                         "runtime_sec": round(self.runtime[model], 2), "rows": len(self.df)})
        fig.suptitle(f"Confusion matrices - {self.name} (rows = true label)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"confusion_{self.name}.png"), dpi=120)
        plt.close(fig)
        return rows

    def agreement_analysis(self):
        """how often do the 3 off-the-shelf models agree, and where don't they?"""
        preds = self.df[[f"pred_{m}" for m in BASE_MODELS]]
        all_agree = preds.nunique(axis=1) == 1
        result = {"dataset": self.name, "all_3_agree": round(all_agree.mean(), 3),
                  # when all agree, are they right? (agreement as a confidence signal)
                  "accuracy_when_all_agree": round((preds[all_agree].iloc[:, 0] == self.df.loc[all_agree, "label"]).mean(), 3)}
        for i, a in enumerate(BASE_MODELS):
            for b in BASE_MODELS[i + 1:]:
                result[f"{a}_vs_{b}"] = round((self.df[f"pred_{a}"] == self.df[f"pred_{b}"]).mean(), 3)
        LOG.info(f"\nAgreement ({self.name}): {result}")

        # Examples for the video: all 3 models give a different answer.
        three_way = self.df[preds.nunique(axis=1) == 3]
        examples = three_way if len(three_way) >= 5 else self.df[~all_agree]
        cols = ["text", "label"] + [f"pred_{m}" for m in BASE_MODELS]
        examples[cols].head(15).to_csv(os.path.join(OUT_DIR, f"disagreements_{self.name}.csv"), index=False)
        return result

    def save_predictions(self):
        self.df.to_csv(os.path.join(OUT_DIR, f"predictions_{self.name}.csv"), index=False)


# ------------------------ cross-dataset steps ---------------------------
# hand-picked sentences that are negative, but VADER reads as neutral
TUNING_EXAMPLES = [ 
    "Kill the process and restart the server.",
    "This bug is nasty, the app crashes on startup.",
    "Fatal error: cannot allocate memory.",
    "The exploit was patched in the latest release.",
    "This method is deprecated, use the new API instead.",
    "Watch out for a deadlock when two threads block on the same lock.",
    "The attack surface is much smaller now.",
    "LGTM, ship it!",
]


def tuning_before_after():
    """show VADER scores before vs after the tech lexicon on clear examples."""
    rows = []
    for text in TUNING_EXAMPLES:
        before = VADER.polarity_scores(text)["compound"]
        after = VADER_TUNED.polarity_scores(clean_dev_text(text))["compound"]
        rows.append({"text": text, "vader_before": before, "label_before": to_label(before, VADER_CUTOFF, -VADER_CUTOFF),
                     "vader_after": after, "label_after": to_label(after, VADER_CUTOFF, -VADER_CUTOFF),
                     "roberta": predict([text], "roberta")[0][0]})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "tuning_examples.csv"), index=False)
    LOG.info("\nTech-lexicon tuning, before vs after:\n" + table.to_string(index=False))
    return table


def plot_model_comparison(summary):
    """Grouped bar chart: macro F1 per model per dataset."""
    pivot = summary.pivot(index="dataset", columns="model", values="macro_f1")[MODELS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        xs = [x + i * width for x in range(len(pivot))]
        bars = ax.bar(xs, pivot[model], width * 0.92, color=MODEL_COLORS[model], label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color="#52514e")
    ax.set_xticks([x + width * (len(MODELS) - 1) / 2 for x in range(len(pivot))], pivot.index)
    ax.set_ylabel("Macro F1 (higher is better)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on developer text")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)


def write_findings(summary, agreement, tuning):
    """Plain-English summary of the run, printed and saved to outputs/findings.md."""
    lines = ["# Findings: can off-the-shelf sentiment models understand developers?", ""]
    lines.append(f"Sample: {SAMPLE_SIZE or 'all'} rows per dataset. Models: {', '.join(MODELS)}.")
    lines.append("")
    lines.append("## Best model per dataset (macro F1)")
    for name, group in summary.groupby("dataset"):
        best = group.sort_values("macro_f1", ascending=False).iloc[0]
        worst = group.sort_values("macro_f1").iloc[0]
        lines.append(f"- **{name}**: best = {best.model} (F1 {best.macro_f1:.2f}, accuracy {best.accuracy:.2f}); "
                     f"worst = {worst.model} (F1 {worst.macro_f1:.2f}).")

    lines += ["", "## Where general-purpose models fail on developer language"]
    for name, group in summary.groupby("dataset"):
        g = group.set_index("model")
        lines.append(f"- **{name}**: VADER calls {g.loc['vader', 'neutral_called_negative']:.0%} of truly neutral posts "
                     f"negative; TextBlob {g.loc['textblob', 'neutral_called_negative']:.0%}; "
                     f"RoBERTa {g.loc['roberta', 'neutral_called_negative']:.0%}.")
    flipped = tuning[tuning.label_before != tuning.label_after]
    lines.append(f"- On {len(tuning)} hand-picked dev sentences, plain VADER labels "
                 f"{(tuning.label_before == 'negative').sum()} as negative; RoBERTa labels "
                 f"{(tuning.roberta == 'negative').sum()} as negative. "
                 f"Example: \"{TUNING_EXAMPLES[0]}\" -> VADER {tuning.vader_before[0]:+.2f}, RoBERTa '{tuning.roberta[0]}'.")

    lines += ["", "## Tech-lexicon tuning (VADER before vs after)"]
    for name, group in summary.groupby("dataset"):
        g = group.set_index("model")
        before, after = g.loc["vader", "macro_f1"], g.loc["vader_tuned", "macro_f1"]
        nb, na = g.loc["vader", "neutral_called_negative"], g.loc["vader_tuned", "neutral_called_negative"]
        lines.append(f"- **{name}**: macro F1 {before:.2f} -> {after:.2f} ({after - before:+.2f}); "
                     f"neutral posts called negative {nb:.0%} -> {na:.0%}.")
    lines.append(f"- On the example sentences, tuning changed {len(flipped)} of {len(tuning)} labels "
                 f"(see outputs/tuning_examples.csv).")

    lines += ["", "## Model agreement (TextBlob, VADER, RoBERTa)"]
    for a in agreement.to_dict("records"):
        lines.append(f"- **{a['dataset']}**: all 3 agree on {a['all_3_agree']:.0%} of posts; when they agree they "
                     f"are right {a['accuracy_when_all_agree']:.0%} of the time. "
                     f"Examples of disagreement: outputs/disagreements_{a['dataset']}.csv.")

    lines += ["", "## Runtime (seconds per dataset sample)"]
    rt = summary.pivot(index="model", columns="dataset", values="runtime_sec")
    for model, row in rt.iterrows():
        lines.append(f"- {model}: " + ", ".join(f"{d} {v:.1f}s" for d, v in row.items()))

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "findings.md"), "w") as f:
        f.write(text + "\n")
    LOG.info("\n" + text)


def main():
    setup_logging()
    start = time.time()
    results, agreement = [], []
    for name, path in DATASETS.items():
        agent = SentimentAgent(name)
        agent.load_data(path)
        agent.compute_sentiment()
        agent.quantitative_sentiment()
        agent.visualize_sentiment()
        results += agent.evaluation_confusion_matrix()
        agreement.append(agent.agreement_analysis())
        agent.save_predictions()

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(os.path.join(OUT_DIR, "agreement_summary.csv"), index=False)
    LOG.info("\n=== Summary (all models, all datasets) ===\n" + summary.to_string(index=False))
    plot_model_comparison(summary)
    tuning = tuning_before_after()

    best_model = summary.groupby("model")["macro_f1"].mean().idxmax()  # best on average
    LOG.info(f"\nBest model overall (mean macro F1): {best_model}")
    write_findings(summary, agreement, tuning)
    LOG.info(f"\nDone in {time.time() - start:.0f}s. Outputs are in {OUT_DIR}/")


if __name__ == "__main__":
    main()
