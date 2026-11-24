import re

import pandas as pd
import streamlit as st

import ui
from registry import BY_KEY

m = BY_KEY["expert_system"]
ui.header(m)
src = ui.load_module(m, "hub_expert_system")

SEVERITY_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
SEVERITY_WEIGHT = {"Critical": 10, "High": 6, "Medium": 3, "Low": 1}
SEVERITY_COLOUR = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "gray"}

advisor = src.NetworkSecurityAdvisor()          # reloads the JSON knowledge base every rerun

tab_assess, tab_acquire, tab_kb = st.tabs(["Assess a network", "Knowledge acquisition", "Knowledge base"])

# ---------------- Interview + inference ----------------
with tab_assess:
    st.markdown("Answer **yes** to everything that is true about the network you are assessing.")
    facts = list(advisor.facts_catalog.items())
    cols = st.columns(2)
    reported = {key for i, (key, q) in enumerate(facts)
                if cols[i % 2].toggle(q, key=f"fact_{key}")}

    advisor.reported_facts = reported
    advisor.infer()                       # original forward-chaining step

    st.divider()
    if not reported:
        st.info("No facts reported yet.")
    elif not advisor.results:
        st.success("No rules fired: no vulnerabilities flagged from the facts reported. "
                   "(This reflects the current rule base only, not a guarantee of security.)")
    else:
        risk = sum(SEVERITY_WEIGHT.get(r["severity"], 1) * r["confidence"] for _, r in advisor.results)
        worst = advisor.results[0][1]["severity"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Vulnerabilities found", len(advisor.results))
        c2.metric("Highest severity", worst)
        c3.metric("Weighted risk score", f"{risk:.1f}", help="Sum of severity weight x confidence "
                  "(Critical 10, High 6, Medium 3, Low 1)")
        st.subheader("Security assessment results")
        for vuln, rule in advisor.results:
            colour = SEVERITY_COLOUR.get(rule["severity"], "gray")
            with st.container(border=True):
                st.markdown(f"#### :{colour}[{rule['severity']}] {vuln}")
                st.caption(f"Confidence {rule['confidence']:.0%} · Why: triggered by "
                           + ", ".join(f"`{f}`" for f in rule["facts"]))
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(rule["recommendation"], 1)))

    # Near misses: rules missing exactly one fact - what to check next
    near = []
    for vuln, rule in advisor.rules.items():
        missing = set(rule["facts"]) - reported
        if len(rule["facts"]) > 1 and len(missing) == 1 and set(rule["facts"]) & reported:
            near.append((vuln, rule, missing.pop()))
    if near:
        st.subheader("One fact away")
        st.caption("Rules that would fire if one more condition were true. Worth checking.")
        for vuln, rule, fact in near:
            st.warning(f"**{vuln}** ({rule['severity']}) also needs: "
                       f"*{advisor.facts_catalog.get(fact, fact)}*")

# ---------------- Knowledge acquisition ----------------
with tab_acquire:
    st.markdown("Add a new rule to the knowledge base (security engineer access). It is saved to "
                "`ExpertSystem/network_security_kb.json` and used immediately.")
    with st.form("new_rule", clear_on_submit=True):
        name = st.text_input("Vulnerability / rule name")
        existing = st.multiselect("Trigger facts (all must be true)", list(advisor.facts_catalog),
                                  format_func=lambda k: f"{k} - {advisor.facts_catalog[k]}")
        st.markdown("New fact (optional)")
        c1, c2 = st.columns([1, 2])
        new_key = c1.text_input("Fact key (short_snake_case)")
        new_q = c2.text_input("Yes/no question to ask for it")
        c1, c2 = st.columns(2)
        severity = c1.selectbox("Severity", ["Low", "Medium", "High", "Critical"], index=2)
        confidence = c2.slider("Confidence this rule indicates real risk", 0.1, 1.0, 0.8, 0.05)
        steps = st.text_area("Hardening / recommendation steps (one per line)")
        add = st.form_submit_button("Add rule", type="primary")
    if add:
        key = re.sub(r"[^a-z0-9_]+", "_", new_key.strip().lower()).strip("_")
        trigger = list(existing) + ([key] if key else [])
        if not name.strip() or name.strip() in advisor.rules:
            st.error("Enter a rule name that is not already in the knowledge base.")
        elif not trigger:
            st.error("Choose at least one trigger fact or create a new one.")
        else:
            if key and key not in advisor.facts_catalog:
                advisor.facts_catalog[key] = new_q.strip() or f"Is '{key}' true?"
            advisor.rules[name.strip()] = {
                "facts": trigger, "severity": severity, "confidence": round(confidence, 2),
                "recommendation": [s.strip() for s in steps.splitlines() if s.strip()] or ["No steps recorded yet."],
            }
            advisor.save_knowledge_base()
            st.success(f"'{name.strip()}' added and saved.")

# ---------------- Knowledge base ----------------
with tab_kb:
    rules = pd.DataFrame([{"Rule": n, "Severity": r["severity"], "Confidence": r["confidence"],
                           "Trigger facts": ", ".join(r["facts"]), "Steps": len(r["recommendation"])}
                          for n, r in advisor.rules.items()])
    rules = rules.sort_values("Severity", key=lambda s: s.map(SEVERITY_RANK), ascending=False)
    st.markdown(f"**{len(advisor.rules)} rules**, **{len(advisor.facts_catalog)} facts**")
    st.dataframe(rules, hide_index=True, width="stretch")
    with st.expander("Facts (questions the advisor asks)"):
        st.dataframe(pd.DataFrame(advisor.facts_catalog.items(), columns=["Fact", "Question"]),
                     hide_index=True, width="stretch")
    c1, c2 = st.columns([2, 1])
    doomed = c1.selectbox("Delete a rule", ["-"] + list(advisor.rules))
    if c2.button("Delete rule", disabled=doomed == "-"):
        del advisor.rules[doomed]
        used = {f for r in advisor.rules.values() for f in r["facts"]}
        advisor.facts_catalog = {k: v for k, v in advisor.facts_catalog.items() if k in used}
        advisor.save_knowledge_base()
        st.rerun()
