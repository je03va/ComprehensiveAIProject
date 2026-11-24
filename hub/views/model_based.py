import random
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import ui
from registry import BY_KEY

m = BY_KEY["model_based"]
ui.header(m)
src = ui.load_module(m, "hub_model_based")   # original file; its run() is not called

DEFAULTS = {"BLOCK_THRESHOLD": 8.0, "ALERT_THRESHOLD": 5.0, "HALF_LIFE_MIN": 15}
ATTACK_IPS = {"203.0.113.50": "brute force", "198.51.100.77": "spraying",
              "192.0.2.99": "takeover", "198.51.100.200": "slow-and-low"}


def simulate(seed):
    """Same perceive -> update model -> decide -> act loop as modelbased.run(), but every
    decision is recorded so the GUI can show it."""
    rng = random.Random(seed)
    day = datetime(2026, 10, 10)
    agent = src.IntrusionAgent()
    for ev in src.generate_history(rng, day):
        agent.learn(ev)
    rows, total = [], 0
    for ev, is_attack, scenario in src.generate_day(rng, day):
        action, reason = agent.step(ev)
        total += src.reward(is_attack, ev.success, action)
        rows.append({"time": ev.ts, "user": ev.user, "ip": ev.ip, "country": ev.country,
                     "success": ev.success, "attack": is_attack, "scenario": scenario,
                     "action": action, "reason": reason, "ip_score": round(agent.ip_score[ev.ip], 2),
                     "cumulative_score": total})
    return agent, pd.DataFrame(rows)


def outcomes(df):
    blocking = df.action.isin(src.BLOCKING_ACTIONS)
    atk, legit = df[df.attack], df[~df.attack]
    return {"Attacks stopped": int(blocking[df.attack].sum()),
            "Breaches": int(((atk.success) & ~atk.action.isin(src.BLOCKING_ACTIONS + ("alert",))).sum()),
            "Missed attempts": int(((~atk.success) & ~atk.action.isin(src.BLOCKING_ACTIONS + ("alert",))).sum()),
            "Legit users blocked": int(blocking[~df.attack].sum()),
            "Unnecessary alerts": int((legit.action == "alert").sum())}


st.subheader("Run a simulated day")
with st.form("mb_form"):
    c1, c2, c3, c4 = st.columns(4)
    block = c1.slider("Block threshold (IP threat score)", 2.0, 15.0, DEFAULTS["BLOCK_THRESHOLD"], 0.5)
    alert = c2.slider("Alert threshold", 1.0, 12.0, DEFAULTS["ALERT_THRESHOLD"], 0.5)
    half_life = c3.slider("Threat score half-life (min)", 1, 60, DEFAULTS["HALF_LIFE_MIN"])
    seed = c4.number_input("Random seed", 0, 9999, 42)
    run = st.form_submit_button("Run agent", type="primary")

if run or "mb" not in st.session_state:
    src.BLOCK_THRESHOLD, src.ALERT_THRESHOLD, src.HALF_LIFE_MIN = block, alert, half_life
    agent, df = simulate(int(seed))
    st.session_state.mb = {"agent": agent, "df": df, "last_ts": df.time.max(),
                           "settings": (block, alert, half_life)}

state = st.session_state.mb
agent, df = state["agent"], state["df"]
src.BLOCK_THRESHOLD, src.ALERT_THRESHOLD, src.HALF_LIFE_MIN = state["settings"]

score = int(df.cumulative_score.iloc[-1])
out = outcomes(df)
cols = st.columns(6)
cols[0].metric("Performance score", score)
for col, (k, v) in zip(cols[1:], out.items()):
    col.metric(k, v)

tab_model, tab_log, tab_test = st.tabs(["Internal model & performance", "Decision log", "Test a login"])

with tab_model:
    left, right = st.columns(2)
    with left:
        st.markdown("**Internal model: IP threat score per attempt** (attack IPs)")
        series = {}
        for ip, label in ATTACK_IPS.items():
            if label == "takeover":
                continue
            s = df[df.ip == ip].ip_score.reset_index(drop=True)
            s.index += 1
            series[f"{label} ({ip})"] = s
        st.line_chart(pd.DataFrame(series), x_label="Attempt number from that IP",
                      y_label="Threat score", height=330)
        st.caption(f"Block threshold = {state['settings'][0]}. Account takeover uses a valid password, "
                   "so it is caught by the user's login profile, not the IP score.")
    with right:
        st.markdown("**Performance measure: cumulative score over the day**")
        st.line_chart(df.set_index("time").cumulative_score, height=330)
    st.markdown("**Outcomes**")
    st.bar_chart(pd.Series(out), horizontal=True, height=230)
    c1, c2 = st.columns(2)
    c1.markdown("**IPs blocked**\n\n" + ("\n".join(f"- `{ip}` {ATTACK_IPS.get(ip, '')}"
                                                    for ip in sorted(agent.blocked_ips)) or "none"))
    c2.markdown("**Accounts locked**\n\n" + ("\n".join(f"- {u}" for u in sorted(agent.locked_users)) or "none"))

with tab_log:
    c1, c2, c3 = st.columns(3)
    actions = c1.multiselect("Action", sorted(df.action.unique()), default=sorted(df.action.unique()))
    scen = c2.multiselect("Scenario", sorted(df.scenario.unique()), default=sorted(df.scenario.unique()))
    only_attacks = c3.toggle("Attacks only")
    view = df[df.action.isin(actions) & df.scenario.isin(scen)]
    if only_attacks:
        view = view[view.attack]
    st.caption(f"{len(view)} of {len(df)} events")
    st.dataframe(view.drop(columns="cumulative_score"), hide_index=True, width="stretch", height=480)

with tab_test:
    st.markdown("Send a login to the agent **as it is now** (after today's events). It updates its "
                "internal model, decides, and acts, exactly like a live event.")
    with st.form("probe"):
        c1, c2, c3 = st.columns(3)
        known = list(src.USERS) + ["(new username)"]
        user = c1.selectbox("User", known)
        if user == "(new username)":
            user = c1.text_input("Username", "admin")
        ip = c2.text_input("IP address", "203.0.113.9")
        country = c3.text_input("Country code", "US")
        c1, c2 = st.columns(2)
        success = c1.radio("Result", ["Failed password", "Successful login"], horizontal=True) == "Successful login"
        minutes = c2.slider("Minutes after the last event", 0, 240, 5)
        go = st.form_submit_button("Send login event", type="primary")
    if go:
        ts = state["last_ts"] + timedelta(minutes=minutes)
        state["last_ts"] = ts
        ev = src.Event(ts, user, ip.strip(), country.strip().upper(), success)
        anomaly = agent.anomaly(ev)
        action, reason = agent.step(ev)
        colour = {"block_ip": "red", "lock_account": "orange", "alert": "orange"}.get(action, "green")
        st.markdown(f"### :{colour}[{action}]")
        st.write(f"Reason: {reason}")
        c1, c2 = st.columns(2)
        c1.markdown("**Internal model after this event**")
        c1.json({**agent.snapshot(ev), "anomaly_for_user (0-4)": anomaly})
        prof = agent.profiles.get(user)
        c2.markdown(f"**Learned profile for {user}**")
        c2.json({k: sorted(v) for k, v in prof.items()} if prof else {"profile": "unknown user"})
        st.session_state.setdefault("probes", []).append(
            {"time": ts, "user": user, "ip": ip, "country": country, "success": success,
             "action": action, "reason": reason})
    if st.session_state.get("probes"):
        st.markdown("**Your test events**")
        st.dataframe(pd.DataFrame(st.session_state.probes).iloc[::-1], hide_index=True, width="stretch")
