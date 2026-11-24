import pandas as pd
import streamlit as st

import launcher
from registry import MODULES, SECTIONS, SUITE_NAME, THEME, module_dir

st.title(SUITE_NAME)
st.markdown(f"One window for every AI product built this semester, all applied to the "
            f"**{THEME}** research theme: keeping devices powered, catching intruders, diagnosing "
            f"AI failures, predicting cloud workloads, advising on network security, and reading "
            f"what developers say about AI tools.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI products", len(MODULES))
c2.metric("Run inside the hub", sum(m["kind"] == "native" for m in MODULES))
c3.metric("Embedded apps", sum(m["kind"] == "streamlit" for m in MODULES))
c4.metric("Desktop games", sum(m["kind"] == "desktop" for m in MODULES))

for section in SECTIONS:
    st.subheader(section)
    mods = [m for m in MODULES if m["section"] == section]
    cols = st.columns(3)
    for i, m in enumerate(mods):
        with cols[i % 3].container(border=True, height=250):
            st.markdown(f"**{m['title']}**")
            st.caption(m["technique"])
            st.write(m["description"][:125].rsplit(" ", 1)[0] + "...")
            st.page_link(f"views/{m['key']}.py", label="Open", icon=m["icon"])

st.divider()
st.subheader("Integration architecture")
st.caption("The hub owns the menu and the title/description panes. Native modules are imported "
           "from their original files; embedded apps run unchanged in their own virtual "
           "environments; games open their own windows.")


def _node(m):
    return f'{m["key"]} [label="{m["title"]}\\n({m["technique"]})"];'


native = "\n".join(_node(m) for m in MODULES if m["kind"] == "native")
embedded = "\n".join(_node(m) for m in MODULES if m["kind"] == "streamlit")
desktop = "\n".join(_node(m) for m in MODULES if m["kind"] == "desktop")
edges = "\n".join(
    f'{"native_hub" if m["kind"] == "native" else "launcher"} -> {m["key"]} '
    f'[label="{ {"native": "import", "streamlit": "subprocess + iframe", "desktop": "subprocess"}[m["kind"]]}"];'
    for m in MODULES)
data = "\n".join(f'{m["key"]} -> {m["key"]}_files [style=dashed, arrowhead=none];\n'
                 f'{m["key"]}_files [label="{m["folder"]}/", shape=folder, fontsize=9, color="#888888"];'
                 for m in MODULES if m["kind"] != "desktop")

st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; nodesep=0.25; ranksep=0.6;
  node [shape=box, style="rounded,filled", fillcolor="#f4f6fb", fontname="Helvetica", fontsize=10];
  edge [fontname="Helvetica", fontsize=8, color="#555555"];
  user [label="User\\n(browser)", shape=ellipse, fillcolor="#fff4d6"];
  hub [label="Hub: hub/app.py\\nStreamlit menu + navigation", fillcolor="#dbe7ff"];
  registry [label="registry.py\\nmodule catalogue", shape=note, fillcolor="#ffffff"];
  native_hub [label="Native pages\\nhub/views/*.py", fillcolor="#e3f4e1"];
  launcher [label="launcher.py\\nprocess manager", fillcolor="#fde7e7"];
  user -> hub; hub -> registry [style=dotted, arrowhead=none]; hub -> native_hub; hub -> launcher;
  subgraph cluster_native {{ label="Imported into the hub"; color="#9fcf9a"; {native} }}
  subgraph cluster_embedded {{ label="Own .venv, own Streamlit server (ports 8601-8604)"; color="#e8a0a0"; {embedded} }}
  subgraph cluster_desktop {{ label="Turtle windows"; color="#c9b3e6"; {desktop} }}
  {edges}
  {data}
}}
""", use_container_width=True)

st.subheader("Module status")
rows = []
for m in MODULES:
    venv = (module_dir(m) / ".venv").exists()
    rows.append({"Module": m["title"], "Section": m["section"], "Technique": m["technique"],
                 "Integration": m["kind"], "Folder": m["folder"],
                 "Own .venv": "yes" if venv else "-",
                 "Process": launcher.status(m) if m["kind"] != "native" else "in hub"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
