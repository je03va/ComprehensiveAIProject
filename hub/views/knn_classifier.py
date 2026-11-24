from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

import ui
from registry import BY_KEY, module_dir

m = BY_KEY["knn_classifier"]
ui.header(m)
src = ui.load_module(m, "hub_learning_agent")    # original learningAgent.py (main() is not run)

FILE = "rag_failure_cases.csv"
folder = module_dir(m)
candidates = [folder / FILE, folder / "data" / FILE, Path(src.DATA_PATH), Path.home() / "Downloads" / FILE]
data_path = next((p for p in candidates if p.exists()), None)

if data_path is None:
    st.warning(f"`{FILE}` was not found. Looked in: " + ", ".join(f"`{p}`" for p in candidates))
    up = st.file_uploader(f"Upload {FILE} (it will be saved into the LearningAgent folder)", type="csv")
    if up is None:
        st.stop()
    data_path = folder / FILE
    data_path.write_bytes(up.getvalue())
    st.rerun()


@st.cache_resource(show_spinner="Learning Element: choosing k by 5-fold cross-validation...")
def train(path, mtime):
    df = pd.read_csv(path)
    train_df, test_df = train_test_split(df, test_size=src.TEST_SIZE, stratify=df["failure_type"],
                                         random_state=src.RANDOM_STATE)
    y_train, y_test = train_df["failure_type"], test_df["failure_type"]
    learners = {kind: src.LearningElement(kind).fit(src.get_inputs(kind, train_df), y_train)
                for kind in ["engineered", "tfidf"]}
    critic = src.Critic(sorted(df["failure_type"].unique()))
    results, preds = {}, {}
    for kind, lr in learners.items():
        preds[kind] = lr.model.predict(src.get_inputs(kind, test_df))
        results[kind] = critic.evaluate(kind, y_test, preds[kind])
    best = max(results, key=lambda k: results[k]["f1"])
    return dict(df=df, train=train_df, test=test_df, learners=learners, results=results,
                preds=preds, best=best, classes=critic.classes)


T = train(str(data_path), data_path.stat().st_mtime)
learners, best = T["learners"], T["best"]
st.caption(f"Data: `{data_path}` · {len(T['df'])} cases · train {len(T['train'])} / test {len(T['test'])}")

tab_diag, tab_learn, tab_critic, tab_pg, tab_saved = st.tabs(
    ["Diagnose a case", "Learning Element", "Critic", "Problem Generator", "Saved report graphs"])

# ---------------- Performance Element ----------------
with tab_diag:
    c1, c2 = st.columns([1, 3])
    kind = c1.radio("Representation", ["engineered", "tfidf"], index=["engineered", "tfidf"].index(best),
                    help=f"Best by test F1: {best}")
    if c2.button("Load a random unseen test case"):
        row = T["test"].sample(1).iloc[0]
        for f in ["question", "retrieved_context", "model_answer", "ground_truth_answer"]:
            st.session_state[f"rag_{f}"] = str(row[f])
        st.session_state.rag_truth = row["failure_type"]
    for f, h in [("question", 70), ("retrieved_context", 120), ("model_answer", 90),
                 ("ground_truth_answer", 70)]:
        st.text_area(f.replace("_", " ").capitalize(), key=f"rag_{f}", height=h)

    if st.button("Diagnose failure", type="primary"):
        case = pd.DataFrame([{f: st.session_state.get(f"rag_{f}", "") for f in
                              ["question", "retrieved_context", "model_answer", "ground_truth_answer"]}])
        if not case.model_answer.iloc[0].strip():
            st.error("Enter at least a model answer.")
        else:
            lr = learners[kind]
            X = src.get_inputs(kind, case)
            label, conf = src.PerformanceElement(lr).predict(X)
            proba = pd.Series(lr.conf_model.predict_proba(X)[0], index=lr.conf_model.classes_)
            left, right = st.columns([1, 1])
            with left:
                st.metric("Predicted failure type", label[0])
                st.metric("Confidence", f"{conf[0]:.0%}")
                if conf[0] < src.LOW_CONF:
                    st.error(f"Low-confidence alert (below {src.LOW_CONF:.0%}): send to human review.")
                else:
                    st.success("Confident diagnosis.")
                truth = st.session_state.get("rag_truth")
                if truth:
                    st.write(f"True label of the loaded case: **{truth}** "
                             + ("(correct)" if truth == label[0] else "(incorrect)"))
                if kind == "engineered":
                    st.markdown("**Features the agent perceived**")
                    st.dataframe(X.T.rename(columns={0: "value"}).round(3))
            with right:
                st.markdown(f"**Neighbour vote (k={lr.conf_model.named_steps['knn'].n_neighbors})**")
                st.bar_chart(proba.sort_values(), horizontal=True, height=300)

# ---------------- Learning Element ----------------
with tab_learn:
    st.markdown("k is **learned from the data** by 5-fold cross-validation on odd k values, "
                "for each way of turning text into numbers.")
    cv = pd.DataFrame({k: lr.cv_scores for k, lr in learners.items()})
    st.line_chart(cv, x_label="k (neighbours)", y_label="CV accuracy", height=320)
    st.dataframe(pd.DataFrame({k: {"best k": lr.best_k, "CV accuracy": round(lr.cv_scores[lr.best_k], 3)}
                               for k, lr in learners.items()}), width="stretch")

# ---------------- Critic ----------------
with tab_critic:
    res = pd.DataFrame(T["results"]).T.round(3)
    st.dataframe(res, width="stretch")
    st.success(f"Best representation by macro F1: **{best}**")
    pick = st.radio("Confusion matrix for", list(learners), index=list(learners).index(best), horizontal=True)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(T["test"]["failure_type"], T["preds"][pick], labels=T["classes"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=T["classes"],
                yticklabels=T["classes"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {pick}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig, width="content")

# ---------------- Problem Generator ----------------
with tab_pg:
    st.markdown("The Problem Generator picks the cases the agent is **least sure about**, so a human "
                "labels those first (active learning).")
    n = st.slider("Cases to suggest", 5, 30, 10)
    X_test = src.get_inputs(best, T["test"])
    perf = src.PerformanceElement(learners[best])
    idx = src.ProblemGenerator.pick(perf, X_test, n)
    labels, conf = perf.predict(X_test)
    picks = T["test"].iloc[idx][["id", "question", "model_answer", "failure_type"]].copy()
    picks.insert(1, "predicted", labels[idx])
    picks.insert(2, "confidence", conf[idx].round(2))
    st.dataframe(picks, hide_index=True, width="stretch")
    alerts = (conf < src.LOW_CONF).sum()
    st.caption(f"{alerts} of {len(conf)} test predictions fall below the {src.LOW_CONF:.0%} review threshold.")

with tab_saved:
    out = folder / src.OUT_DIR
    imgs = sorted(out.glob("*.png")) if out.exists() else []
    if not imgs:
        st.info("No saved graphs yet. Run `python learningAgent.py` in the LearningAgent folder to create them.")
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        cols[i % 2].image(str(p), caption=p.name)
