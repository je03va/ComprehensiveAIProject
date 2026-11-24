# IT Ops AI Suite (integrated menu GUI)

One Streamlit window that opens all 11 AI products from this semester, grouped by section
like an office suite. Every module has its own title/description pane (technique, PEAS,
and what was improved for the suite).

## Run it

From the `AIProjects` folder:

```bash
./hub/run_hub.sh
```

The first run creates `hub/.venv` and installs `hub/requirements.txt`. After that it opens
the suite at http://localhost:8600.

## How the modules are integrated

| Integration | Modules | How |
|---|---|---|
| **Native page** (runs inside the hub) | Battery Swap Reflex Agent, Login Intrusion Monitor, RAG Failure Diagnosis Agent, Network Security Advisor, Article Q&A Chatbot | The hub imports the classes from the original files (`SimpleReflexAgent/simple_reflex_agent_it.py`, `ModelBasedAgent/modelbased.py`, `LearningAgent/learningAgent.py`, `ExpertSystem/expert_system.py`, `src/article_qa/`) and adds a GUI around them. The original files are unchanged. |
| **Embedded app** (own process, own `.venv`) | Transcode Time Predictor (kNN), MLR vs kNN, Corpus NLP Monitor, Developer Sentiment Analyzer | `launcher.py` starts the module's existing Streamlit app with **its own `.venv`** on ports 8601-8604, with the module folder as its working directory, and the hub shows it in an iframe. Separate environments are needed because NLPSentiment needs NumPy 1.x while the regression modules use NumPy 2.x. |
| **Desktop window** | Data Structures lesson, Packet Race | The turtle game opens in its own window. |

## Files

```
hub/
  app.py          entry point: sidebar menu (st.navigation)
  registry.py     catalogue of every module: section, technique, PEAS, description, integration type
  launcher.py     starts/stops embedded apps and games, picks each module's .venv
  ui.py           title/description pane, embedded-app page, game page, load-by-path helper
  views/          one page per module + home.py (overview, architecture diagram, status)
ExpertSystem/
  expert_system.py           moved here from csc 470/expertSystemAssignment.ipynb
  network_security_kb.json   knowledge base (test rules "hi" / "jalen" removed)
```

To add a module: add an entry to `registry.py` and a page in `views/` named after its key.

## Data notes

- **RAG Failure Diagnosis:** looks for `rag_failure_cases.csv` in `LearningAgent/`, then
  `~/Downloads/`. If it isn't in either place, the page lets you upload it.
- **Article Q&A:** reads `GEMINI_API_KEY` from `AIProjects/.env`. You can also paste the key
  into the page's settings.
- **Games:** need Python with Tk (python.org Python has it; with Homebrew run `brew install python-tk`).
- Embedded apps read their data exactly as they do when run on their own. The kNN regressor
  uses the transcoding TSV in `KnnMLR/data/`, and falls back to `~/Downloads/`.
