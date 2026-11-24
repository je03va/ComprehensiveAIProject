import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings
DATA_PATH = "/Users/je03va/Downloads/rag_failure_cases.csv"
OUT_DIR = "agent_output"           # graphs and the prediction log are saved here
RANDOM_STATE = 42
TEST_SIZE = 0.20
K_VALUES = list(range(1, 32, 2))   # odd k only, avoids vote ties
CV_FOLDS = 5
LOW_CONF = 0.60                    # below this confidence gives alert
CONF_MIN_K = 5                     # confidence needs >= 5 neighbours (k=1 gives only 0/1)


def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def save_and_show(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.show()
    plt.close(fig)


# Data and percepts
# severity, evidence_quality and explanation are not used as inputs: they are
# determined by the label, so the agent would not know them at prediction time.
def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"Could not find {path}. Update DATA_PATH at the top of this file.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns. Missing values: {int(df.isna().sum().sum())}")
    print("\nClass balance:\n" + df["failure_type"].value_counts().to_string())
    return df


def tokens(text):
    """Set of words in a text (lowercase, ignores punctuation and repeats)."""
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def jaccard(a, b):
    """Share of words two texts have in common (ignores order and repeats)."""
    A, B = tokens(a), tokens(b)
    return len(A & B) / max(len(A | B), 1)


def build_features(df):
    """Engineered features: relationships between the text fields."""
    X = pd.DataFrame(index=df.index)
    X["ans_ctx_overlap"] = [jaccard(a, c) for a, c in zip(df.model_answer, df.retrieved_context)]
    X["ans_gt_overlap"] = [jaccard(a, g) for a, g in zip(df.model_answer, df.ground_truth_answer)]
    X["q_ctx_overlap"] = [jaccard(q, c) for q, c in zip(df.question, df.retrieved_context)]
    X["len_ans"] = df.model_answer.str.split().str.len()
    X["len_gt"] = df.ground_truth_answer.str.split().str.len()
    X["len_ratio"] = X["len_ans"] / X["len_gt"].clip(lower=1)
    return X


def get_inputs(kind, df):
    """Two ways to turn a case into numbers: engineered features or TF-IDF of the answer."""
    return build_features(df) if kind == "engineered" else df["model_answer"].astype(str)


def make_pipeline(kind, k):
    """Scaler (or TF-IDF) + kNN with Euclidean distance."""
    first = (("scale", StandardScaler()) if kind == "engineered"
             else ("tfidf", TfidfVectorizer()))
    return Pipeline([first, ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))])


# Learning agent components
class LearningElement:
    """Learns the neighbour memory and learns k from data by cross-validation."""

    def __init__(self, kind):
        self.kind = kind
        self.best_k = None
        self.cv_scores = {}
        self.model = None        # predicts the label (uses best_k)
        self.conf_model = None   # same data, k >= CONF_MIN_K, used for confidence

    def select_k(self, X, y):
        """Selects the best k by cross-validation, stores the scores in self.cv_scores."""
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for k in K_VALUES:
            self.cv_scores[k] = cross_val_score(make_pipeline(self.kind, k), X, y, cv=cv).mean()
        return max(self.cv_scores, key=self.cv_scores.get)

    def fit(self, X, y, k=None):
        """Fits the model to the data, selecting k if not provided."""
        self.best_k = k if k is not None else self.select_k(X, y)
        self.model = make_pipeline(self.kind, self.best_k).fit(X, y)
        conf_k = max(self.best_k, CONF_MIN_K)
        self.conf_model = clone(self.model).set_params(knn__n_neighbors=conf_k).fit(X, y)
        return self


class PerformanceElement:
    """Diagnoses a case, reports confidence, alerts if unsure, logs the result."""

    def __init__(self, learner):
        self.learner = learner

    def predict(self, X):
        """Predicts the label and confidence for each case."""
        labels = self.learner.model.predict(X)
        proba = self.learner.conf_model.predict_proba(X)
        classes = list(self.learner.conf_model.classes_)
        conf = np.array([proba[i, classes.index(lab)] for i, lab in enumerate(labels)])
        return labels, conf

    def predict_and_log(self, X, ids, y_true):
        """Predicts, logs the results and alerts if confidence is low."""
        labels, conf = self.predict(X)
        log = pd.DataFrame({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "id": list(ids), "predicted": labels, "actual": list(y_true),
            "confidence": conf.round(3), "low_conf_alert": conf < LOW_CONF})
        log["correct"] = log["predicted"] == log["actual"]
        log_path = os.path.join(OUT_DIR, "prediction_log.csv")
        log.to_csv(log_path, index=False)
        print(f"[log] wrote {len(log)} predictions to {log_path}")
        alerts = log[log.low_conf_alert]
        print(f"[alert] {len(alerts)} of {len(log)} predictions below confidence {LOW_CONF:.2f}")
        for _, r in alerts.head(5).iterrows():
            print(f"   ALERT id={r['id']}: predicted '{r['predicted']}' "
                  f"(confidence {r['confidence']:.2f}) -> send to human review")
        return labels


class Critic:
    """Measures performance against the true labels."""

    def __init__(self, classes):
        self.classes = list(classes)

    def evaluate(self, name, y_true, y_pred):
        m = {"accuracy": accuracy_score(y_true, y_pred),
             "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
             "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
             "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}
        print(f"\n--- Critic report: {name} ---")
        print(f"accuracy {m['accuracy']:.3f} | precision {m['precision']:.3f} | "
              f"recall {m['recall']:.3f} | F1 {m['f1']:.3f}  (macro averages)")
        return m

    def report_and_plot(self, name, y_true, y_pred):
        print(classification_report(y_true, y_pred, labels=self.classes, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.classes, yticklabels=self.classes, ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {name}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        save_and_show(fig, "confusion_matrix.png")


class ProblemGenerator:
    """Picks the unlabeled cases the agent is least confident about."""

    @staticmethod
    def pick(performance, X_pool, n):
        _, conf = performance.predict(X_pool)
        return np.argsort(conf)[:n]


# Experiments
def plot_accuracy_vs_k(learners):
    """Plot the cross-validation accuracy for each representation, mark the best k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, lr in learners.items():
        ax.plot(list(lr.cv_scores), list(lr.cv_scores.values()), marker="o", label=kind)
        ax.scatter([lr.best_k], [lr.cv_scores[lr.best_k]], s=160, facecolors="none",
                   edgecolors="red", linewidths=2)
    ax.set(xlabel="k (number of neighbours)", ylabel=f"{CV_FOLDS}-fold CV accuracy",
           title="Choosing k (red circle = selected k)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_and_show(fig, "accuracy_vs_k.png")


def run_active_learning(kind, train, y_train, test, y_test, strategy,
                        seed_size=40, batch=20, rounds=15, k=5):
    """Start with few labels. Each round: pick cases, a 'human' labels them
    , the agent retrains. Returns (n_labelled, accuracy)."""
    rng = np.random.RandomState(RANDOM_STATE)
    X_train, X_test = get_inputs(kind, train), get_inputs(kind, test)
    labelled, _ = train_test_split(np.arange(len(train)), train_size=seed_size,
                                   stratify=y_train, random_state=RANDOM_STATE)
    labelled = list(labelled)
    pool = [i for i in range(len(train)) if i not in labelled]
    curve = []
    for r in range(rounds + 1):
        lr = LearningElement(kind).fit(X_train.iloc[labelled], y_train.iloc[labelled], k=k)
        curve.append((len(labelled), accuracy_score(y_test, lr.model.predict(X_test))))
        if r == rounds:
            break
        if strategy == "uncertainty":
            picks = ProblemGenerator.pick(PerformanceElement(lr), X_train.iloc[pool], batch)
        else:
            picks = rng.choice(len(pool), size=batch, replace=False)
        chosen = [pool[p] for p in picks]
        labelled += chosen
        pool = [i for i in pool if i not in chosen]
    return curve


def learning_curve_experiment(kind, train, y_train, test, y_test):
    banner(f"IMPROVEMENT: the agent learns from feedback ({kind}, k=5)")
    unc = run_active_learning(kind, train, y_train, test, y_test, "uncertainty")
    rnd = run_active_learning(kind, train, y_train, test, y_test, "random")
    print(f"{'labelled':>9} | {'uncertainty':>11} | {'random':>6}")
    for (n, a), (_, b) in list(zip(unc, rnd))[::3]:
        print(f"{n:>9} | {a:>11.3f} | {b:>6.3f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*unc), marker="o", label="Problem Generator (least confident first)")
    ax.plot(*zip(*rnd), marker="s", label="random labelling")
    ax.set(xlabel="number of labelled examples", ylabel="test accuracy",
           title="Learning curve: agent improves as it receives labels")
    ax.legend()
    ax.grid(alpha=0.3)
    save_and_show(fig, "learning_curve.png")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    banner("1. PERCEPTS: load the data")
    df = load_data(DATA_PATH)

    banner("2. PREPROCESSING: stratified train/test split")
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["failure_type"],
                                   random_state=RANDOM_STATE)
    y_train, y_test = train["failure_type"], test["failure_type"]
    print(f"train {len(train)} rows | test {len(test)} rows")

    banner("3. LEARNING: choose k by cross-validation for each representation")
    learners = {}
    for kind in ["engineered", "tfidf"]:
        learners[kind] = LearningElement(kind).fit(get_inputs(kind, train), y_train)
        lr = learners[kind]
        print(f"{kind:>10}: best k = {lr.best_k} (CV accuracy {lr.cv_scores[lr.best_k]:.3f})")
    plot_accuracy_vs_k(learners)

    banner("4. PERFORMANCE + CRITIC: predict on unseen test data")
    critic = Critic(sorted(df["failure_type"].unique()))
    results, predictions = {}, {}
    for kind, lr in learners.items():
        predictions[kind] = lr.model.predict(get_inputs(kind, test))
        results[kind] = critic.evaluate(kind, y_test, predictions[kind])
    best_kind = max(results, key=lambda kd: results[kd]["f1"])
    print(f"\nBest representation by F1: {best_kind}")

    # Full report, confusion matrix, log and alerts for the best agent
    perf = PerformanceElement(learners[best_kind])
    labels = perf.predict_and_log(get_inputs(best_kind, test), test["id"], y_test)
    critic.report_and_plot(best_kind, y_test, labels)

    learning_curve_experiment(best_kind, train, y_train, test, y_test)

    banner(f"DONE - graphs and prediction_log.csv saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()