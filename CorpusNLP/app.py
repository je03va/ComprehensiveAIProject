"""
INNOVATION 1: Streamlit dashboard for the Corpus Analysis NLP Agent.
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import agent

st.set_page_config(page_title="AI Coding Assistant Monitor", layout="wide")
st.title("AI Coding Assistant Discussion Monitor")
st.caption("Sentiment, emerging topics and risk indicators across 6 public sources")


# Cache so changing a filter does not re-download or re-score everything
@st.cache_data
def load_corpus():
    return agent.CorpusAgent().load_corpus(agent.ALL_SOURCES)


@st.cache_data
def run_analysis(df):
    return agent.analyze(df)


# ---------------- Sidebar: source picker and time window ----------------
st.sidebar.header("Settings")
if st.sidebar.button("Refresh data from the web (slow)"):
    agent.CorpusAgent().load_corpus(agent.ALL_SOURCES, refresh=True)
    st.cache_data.clear()

corpus = load_corpus()
sources = st.sidebar.multiselect("Sources", agent.ALL_SOURCES, default=agent.ALL_SOURCES)
months = [str(p) for p in pd.period_range(agent.START_MONTH, agent.END_MONTH, freq="M")]
start, end = st.sidebar.select_slider("Time window", options=months, value=(months[0], months[-1]))
st.sidebar.caption(f"Alert thresholds: sentiment drop > {agent.SENTIMENT_DROP_THRESHOLD}, "
                   f"risk spike > {agent.RISK_SPIKE_PCT}% (min {agent.MIN_RISK_DOCS} docs)")

in_window = (corpus["date"] >= start) & (corpus["date"] < pd.Period(end, "M").end_time)
df = corpus[corpus["source"].isin(sources) & (in_window | (corpus["source"] == "gutenberg"))]
if df.empty:
    st.warning("No documents for this selection.")
    st.stop()

results = run_analysis(df)
docs = results["docs"]

# ---------------- Headline numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(docs))
c2.metric("Sources", docs["source"].nunique())
c3.metric("Mean VADER (trend sources)", f"{results['overall']['vader'].mean():+.3f}")
c4.metric("Alerts fired", len(results["alerts"]))

# ---------------- Alerts panel ----------------
st.subheader("Alerts")
if results["alerts"].empty:
    st.success("No alerts for this selection.")
for _, a in results["alerts"].iloc[::-1].iterrows():   # newest first
    message = f"**{a['period']} - {a['type'].replace('_', ' ')} [{a['item']}]**: {a['detail']}"
    if a["type"] == "emerging_topic":
        st.info(message)
    else:
        st.error(message)
    with st.expander("Example text"):
        st.write(a["example"])

# ---------------- Sentiment over time ----------------
st.subheader("Sentiment over time")
left, right = st.columns(2)
left.caption("Mean score per month (VADER vs TextBlob)")
left.line_chart(results["overall"][["vader", "textblob"]])
right.caption("Mean VADER per month, by source")
right.line_chart(results["by_source"])

# ---------------- Risk indicators and assistants ----------------
left, right = st.columns(2)
left.subheader("Risk indicators")
left.caption("% of documents mentioning each risk category")
left.line_chart(results["risk_rates"])
right.subheader("Assistant mentions")
right.caption("% of documents mentioning each assistant")
right.line_chart(results["tool_rates"])

# ---------------- Keywords and emerging topics ----------------
left, right = st.columns(2)
left.subheader("Top keywords")
left.bar_chart(results["keywords"].set_index("keyword"), horizontal=True)
right.subheader("Emerging topics (latest month vs earlier)")
right.dataframe(results["emerging"], hide_index=True)

# ---------------- Example texts ----------------
st.subheader("Example texts")
source_pick = st.selectbox("Source", sorted(docs["source"].unique()))
subset = docs[docs["source"] == source_pick].sort_values("vader")
left, right = st.columns(2)
left.markdown("**Most negative**")
left.dataframe(subset.head(5)[["date", "vader", "text", "url"]], hide_index=True)
right.markdown("**Most positive**")
right.dataframe(subset.tail(5).iloc[::-1][["date", "vader", "text", "url"]], hide_index=True)

# ---------------- Evaluation ----------------
st.subheader("Evaluation: VADER vs TextBlob on the same documents")
st.dataframe(results["evaluation"], hide_index=True)
st.caption("Change the sources or time window in the sidebar to rerun the comparison.")
