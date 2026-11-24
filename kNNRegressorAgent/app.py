"""Streamlit GUI: run with `streamlit run app.py`."""
import pandas as pd
import streamlit as st

from agent import (CATEGORICAL, LOG_DIR, MODEL_PATH, NUMERIC, LearningAgent, clean_percept, load_data,
                   log_csv, plot_actual_vs_predicted, plot_cv_curves, plot_residuals)

st.set_page_config(page_title="Transcode Time Agent", layout="wide")
st.title("Video Transcoding Time Agent (kNN)")


@st.cache_resource
def get_agent():
    if MODEL_PATH.exists():
        return LearningAgent.load()
    with st.spinner("No saved model found - training (takes a few minutes)..."):
        agent = LearningAgent().train(load_data())
        agent.save()
    return agent


agent = get_agent()
performer = agent.performer
defaults = load_data()[NUMERIC].median()  # a typical job as the form's starting values

FORM_LAYOUT = {
    "Input video": ["duration", "codec", "width", "height", "bitrate", "framerate"],
    "Frame statistics": ["i", "p", "b", "frames", "i_size", "p_size", "b_size", "size"],
    "Output settings": ["o_codec", "o_bitrate", "o_framerate", "o_width", "o_height"],
}

predict_tab, training_tab, eval_tab, explore_tab = st.tabs(
    ["Predict", "Training", "Evaluation", "Problem Generator"])

with predict_tab:
    with st.form("job"):
        raw = {}
        for section, columns in zip(FORM_LAYOUT, st.columns(len(FORM_LAYOUT))):
            columns.subheader(section)
            for col in FORM_LAYOUT[section]:
                if col in CATEGORICAL:
                    raw[col] = columns.selectbox(col, sorted(performer.known_codecs[col]))
                else:  # free text on purpose, so bad/missing values can be tested
                    raw[col] = columns.text_input(col, f"{defaults[col]:g}")
        submitted = st.form_submit_button("Predict transcoding time")

    if submitted:
        job, warnings = clean_percept(raw)
        result = performer.predict_job(job)
        for warning in warnings:
            st.warning(warning)

        left, right = st.columns(2)
        left.metric("Predicted utime", f"{result['prediction']:.2f} s")
        right.metric("Confidence", result["confidence"], f"neighbor spread {result['spread']:.2f}",
                     delta_color="off")
        for alert in result["alerts"]:
            st.error(f"Low-confidence alert: {alert}")
        st.caption("utime of the nearest training jobs: "
                   + ", ".join(f"{t:.2f}s" for t in result["neighbor_times"]))

        log_csv("predictions.csv", {**job, "prediction": result["prediction"],
                                    "spread": result["spread"], "confidence": result["confidence"],
                                    "alerts": " | ".join(result["alerts"] + warnings)})

    log_path = LOG_DIR / "predictions.csv"
    if log_path.exists():
        with st.expander("Prediction log"):
            st.dataframe(pd.read_csv(log_path).tail(20))

with training_tab:
    st.subheader("Selected by cross-validation")
    st.write(agent.learner.best_params)
    st.pyplot(plot_cv_curves(agent.learner.cv_results))
    st.subheader("All feature sets vs. trimmed (best config each)")
    cv_results = agent.learner.cv_results
    st.dataframe(cv_results.sort_values("R2").groupby("features").tail(1), hide_index=True)
    with st.expander("Full cross-validation results"):
        st.dataframe(cv_results, hide_index=True)

with eval_tab:
    st.subheader("Test set: tuned agent vs. class baseline (k=5, single split)")
    st.dataframe(pd.DataFrame({"baseline": agent.baseline_metrics, "tuned": agent.metrics}).round(3))
    st.write(f"Critic flagged **{agent.high_error.sum()}** of {len(agent.high_error)} test jobs "
             f"with absolute error above {agent.error_threshold:.2f}s.")
    left, right = st.columns([1, 2])
    left.pyplot(plot_actual_vs_predicted(agent.y_test, agent.y_pred))
    right.pyplot(plot_residuals(agent.y_test, agent.y_pred))

with explore_tab:
    st.subheader("Job types to test next")
    st.caption("Codec/resolution combinations with few training samples or high test error.")
    st.dataframe(agent.suggestions.round(2), hide_index=True)
