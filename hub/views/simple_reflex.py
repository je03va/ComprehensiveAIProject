import random
import sqlite3
import time

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import streamlit as st

import ui
from registry import BY_KEY

m = BY_KEY["simple_reflex"]
ui.header(m)

# Load the original agent class; the driver loop at the bottom of the script is left out
src = ui.load_module(m, "hub_simple_reflex", strip_from="\nagent = SimpleReflexAgent()")


class HubReflexAgent(src.SimpleReflexAgent):
    """The original agent, with an in-memory database and no pop-up matplotlib window.
    environment(), perception_decision() and performance_measure() are the original methods."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE batteries (name TEXT PRIMARY KEY, type TEXT, charge INTEGER, max_drain INTEGER);
            CREATE TABLE log (step INTEGER, battery TEXT, charge INTEGER, action TEXT);""")
        self.db.executemany("INSERT INTO batteries VALUES (?, ?, 100, ?)", src.DEVICES)
        self.step = 0
        self.action = {}

    def log_frame(self):
        return pd.read_sql("SELECT * FROM log", self.db)


def simulate(threshold, steps, seed):
    src.THRESHOLD = threshold          # the rule reads this module-level constant
    random.seed(seed)
    agent = HubReflexAgent()
    for _ in range(steps):
        agent.environment()
        agent.perception_decision()
    return agent


def summary_table(agent):
    return pd.read_sql("""SELECT battery AS Device, SUM(action='replace') AS Swaps, SUM(charge=0) AS Outages,
                          MIN(charge) AS "Lowest %",
                          ROUND(AVG(CASE WHEN action='replace' THEN charge END), 1) AS "Avg % at swap"
                          FROM log GROUP BY battery""", agent.db)


st.subheader("Run the agent")
with st.form("reflex_form"):
    c1, c2, c3, c4 = st.columns(4)
    threshold = c1.slider("Swap threshold (%)", 1, 60, src.THRESHOLD)
    steps = c2.slider("Time steps", 20, 300, src.STEPS, step=10)
    seed = c3.number_input("Random seed", 0, 9999, 7)
    animate = c4.toggle("Animate", value=True)
    run = st.form_submit_button("Run simulation", type="primary")

if run:
    src.THRESHOLD = threshold
    random.seed(seed)
    agent = HubReflexAgent()
    chart_slot, status_slot = st.empty(), st.empty()
    for i in range(steps):
        agent.environment()
        agent.perception_decision()
        if animate and (i % 3 == 0 or i == steps - 1):
            log = agent.log_frame()
            recent = log[log.step > agent.step - 60].pivot(index="step", columns="battery", values="charge")
            chart_slot.line_chart(recent, y_label="Battery %", height=320)
            score, outages, _ = agent.performance_measure()
            status_slot.caption(f"Step {agent.step} · outages {outages} · score {score:.1f}")
            time.sleep(0.03)
    st.session_state.reflex = {"agent": agent, "threshold": threshold}
    chart_slot.empty()
    status_slot.empty()

result = st.session_state.get("reflex")
if result:
    agent = result["agent"]
    score, outages, avg = agent.performance_measure()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance score", f"{score:.1f}")
    c2.metric("Outages (hit 0%)", outages)
    c3.metric("Avg charge left at swap", f"{avg:.1f}%")
    c4.metric("Threshold used", f"{result['threshold']}%")

    log = agent.log_frame()
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Battery charge per device** (dips below the threshold trigger a swap)")
        st.line_chart(log.pivot(index="step", columns="battery", values="charge"),
                      y_label="Battery %", x_label="Time step", height=340)
    with right:
        table = summary_table(agent)
        st.markdown("**Battery swaps per device**")
        st.bar_chart(table.set_index("Device")["Swaps"], height=340, color="#2e9e5b")
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Percept / action log (SQLite `log` table)"):
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")

st.divider()
st.subheader("Optimization: which threshold scores best?")
st.caption("Runs the same agent for every threshold over several random seeds and averages the "
           "performance measure. Too low causes outages; too high wastes charge at every swap.")
c1, c2, c3 = st.columns([1, 1, 2])
n_seeds = c1.slider("Seeds per threshold", 3, 30, 10)
sweep_steps = c2.slider("Steps per run", 50, 300, 100, step=10)
if c3.button("Run threshold sweep"):
    rows = []
    bar = st.progress(0.0)
    thresholds = list(range(2, 52, 2))
    for i, t in enumerate(thresholds):
        for s in range(n_seeds):
            score, outages, avg = simulate(t, sweep_steps, s).performance_measure()
            rows.append({"threshold": t, "score": score, "outages": outages, "avg_at_swap": avg})
        bar.progress((i + 1) / len(thresholds))
    bar.empty()
    src.THRESHOLD = 15
    st.session_state.sweep = pd.DataFrame(rows).groupby("threshold").mean()

sweep = st.session_state.get("sweep")
if sweep is not None:
    best = sweep["score"].idxmax()
    st.success(f"Best average score **{sweep.loc[best, 'score']:.1f}** at a **{best}%** threshold "
               f"({sweep.loc[best, 'outages']:.1f} outages, {sweep.loc[best, 'avg_at_swap']:.1f}% "
               f"left at swap on average).")
    left, right = st.columns(2)
    left.markdown("**Average performance score**")
    left.line_chart(sweep["score"], x_label="Threshold %", height=260)
    right.markdown("**Outages vs charge wasted at swap**")
    right.line_chart(sweep[["outages", "avg_at_swap"]], x_label="Threshold %", height=260)
