# Handoff report: IT Ops AI Suite (hub)

Context for continuing work on this project. Written 2026-11-23.

## What the project is

A single Streamlit app ("IT Ops AI Suite") that opens 11 AI course projects from one sidebar menu, grouped into four sections. Each module page starts with a title/description pane showing its technique, PEAS (Performance, Environment, Actuators, Sensors), and the improvements made for the suite.

- Project root: `/Users/je03va/Desktop/AIProjects/`
- Hub folder: `/Users/je03va/Desktop/AIProjects/hub/`
- Hub URL when running: http://localhost:8600

## Hub files

| File | Role |
|---|---|
| `app.py` | Entry point. Builds the sidebar menu with `st.navigation`. |
| `registry.py` | Catalogue of every module: key, section, title, technique, integration kind, folder, entry file, port, PEAS, enhancements. `ROOT` = AIProjects/. |
| `launcher.py` | Starts/stops embedded Streamlit apps and desktop games. Picks each module's own `.venv` python. Child processes run with the module folder as working directory. |
| `ui.py` | Shared UI: description pane, embedded-app page (iframe), game page, load-module-by-path helper. |
| `views/` | One page per module, named after its registry `key`, plus `home.py` (overview, architecture diagram, status). |
| `requirements.txt` | Hub environment: streamlit>=1.50, pandas, numpy, matplotlib, seaborn, scikit-learn, python-dotenv, pypdf, gTTS, google-genai. |
| `run_hub.sh` | Launcher script. Creates `hub/.venv` on first run, then runs `streamlit run hub/app.py --server.port 8600`. |

To add a module: add an entry to `registry.py` and a page in `views/` named after its key.

## Modules and how they're integrated

**Native pages** (run inside the hub process, using `hub/.venv`). The hub imports classes from the original files, which are left unchanged.

| Module | Source |
|---|---|
| Battery Swap Reflex Agent (simple reflex) | `SimpleReflexAgent/simple_reflex_agent_it.py` |
| Login Intrusion Monitor (model-based reflex) | `ModelBasedAgent/modelbased.py` |
| RAG Failure Diagnosis Agent (kNN classifier) | `LearningAgent/learningAgent.py` |
| Network Security Advisor (forward-chaining expert system) | `ExpertSystem/expert_system.py` + `network_security_kb.json` |
| Article Q&A Chatbot (Gemini + TTS) | `src/article_qa/` |

**Embedded apps** (each runs as its own Streamlit server with its own `.venv`, shown in an iframe). These are separate because NLPSentiment needs NumPy 1.x while the regression modules use NumPy 2.x.

| Module | Folder | Entry | Port |
|---|---|---|---|
| Transcode Time Predictor (kNN regressor) | `kNNRegressorAgent` | `app.py` | 8601 |
| MLR vs kNN Learning Agent | `KnnMLR` | `app.py` | 8602 |
| AI Coding Assistant Discussion Monitor (corpus NLP) | `CorpusNLP` | `app.py` | 8603 |
| Developer Sentiment Analyzer | `NLPSentiment` | `dashboard.py` | 8604 |

**Desktop windows** (turtle graphics, need Python with Tk): Data Structures lesson (`GameProject/data_structures_lesson.py`) and Packet Race BFS vs DFS vs A* (`GameProject/search_race.py`).

## Current environment state (checked 2026-11-23)

- `hub/.venv` created with Python 3.13.9 (python.org install at `/Library/Frameworks/Python.framework/Versions/3.13/`). All of `requirements.txt` is installed; a test import worked; Streamlit is 1.67.0. `.venv/` is in `hub/.gitignore`.
- All four embedded apps already have their own `.venv`.
- `AIProjects/.env` exists (the chatbot reads `GEMINI_API_KEY` from it).
- `rag_failure_cases.csv` is in `~/Downloads/` (not in `LearningAgent/`). The RAG page checks `LearningAgent/` first, then `~/Downloads/`, so it will find it.
- `hub/run_hub.sh` had no execute permission, which gave "permission denied". Fixed with `chmod +x`.

## How to run

```zsh
cd /Users/je03va/Desktop/AIProjects/hub
./run_hub.sh
```

To activate the hub venv in a terminal (it must be sourced, not run directly, or you get "permission denied"):

```zsh
source /Users/je03va/Desktop/AIProjects/hub/.venv/bin/activate
```

## Git state

- Branch `main`. Recent commits are for the Article Q&A chatbot (source-grounded prompt, TTS playback, README).
- The whole `hub/` folder is untracked, as are most module folders (CorpusNLP, ExpertSystem, GameProject, KnnMLR, LearningAgent, ModelBasedAgent, NLPSentiment, SimpleReflexAgent, kNNRegressorAgent). `AIProjects/.gitignore` has uncommitted changes. Nothing has been committed for the hub yet.

## Open items

- Hub not yet run end to end in this session; no module pages have been tested since the venv was created.
- Hub and module folders still need committing if they should be in git.
