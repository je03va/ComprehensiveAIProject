# IT Ops AI Suite

The AI projects I built this semester, each framed as an IT operations tool, plus a
Streamlit **hub** (`hub/`) that opens all 11 of them from one menu, grouped like an office suite.

| Section | Module | Technique | Folder |
|---|---|---|---|
| Intelligent Agents | Battery Swap Reflex Agent | Simple reflex agent | `SimpleReflexAgent/` |
| Intelligent Agents | Login Intrusion Monitor | Model-based reflex agent | `ModelBasedAgent/` |
| Learning Agents | RAG Failure Diagnosis Agent | kNN classifier | `LearningAgent/` |
| Learning Agents | Transcode Time Predictor | kNN regressor | `kNNRegressorAgent/` |
| Learning Agents | MLR vs kNN Learning Agent | Multilinear regression | `KnnMLR/` |
| Knowledge & Language | Network Security Advisor | Expert system (forward chaining) | `ExpertSystem/` |
| Knowledge & Language | Article Q&A Chatbot | Gemini LLM + TTS | `src/article_qa/` |
| Knowledge & Language | AI Coding Assistant Discussion Monitor | NLP corpus analysis | `CorpusNLP/` |
| Knowledge & Language | Developer Sentiment Analyzer | Comparative sentiment | `NLPSentiment/` |
| Games & Simulation | Data Structures Behind Your IT | Animated game-like agent | `GameProject/` |
| Games & Simulation | Packet Race: BFS vs DFS vs A* | Search simulation | `GameProject/` |

## Getting started

### Requirements

- **macOS or Linux** with `git` and **Python 3.12**. Python 3.13 works for the hub itself,
  but the Sentiment module's PyTorch pin has no 3.13 build.
- **Tk**, for the two desktop games. python.org installers include it. With Homebrew, run
  `brew install python-tk@3.12`.
- **A Gemini API key** for the Article Q&A chatbot. It's free from
  [Google AI Studio](https://aistudio.google.com/apikey). The other modules work without it.

### 1. Clone

```bash
git clone https://github.com/je03va/ComprehensiveAIProject.git
cd ComprehensiveAIProject
```

### 2. Add your API key

```bash
cp .env.example .env
# open .env and set GEMINI_API_KEY=...
```

### 3. Set up the embedded modules (one time)

Four modules (kNN regressor, MLR, Corpus NLP, Sentiment) run as separate Streamlit apps
that the hub embeds. Each needs its own virtual environment because they require different
NumPy versions:

```bash
for d in kNNRegressorAgent KnnMLR CorpusNLP NLPSentiment; do
  python3.12 -m venv $d/.venv
  $d/.venv/bin/pip install -r $d/requirements.txt
done
```

This takes several minutes, mostly for PyTorch and transformers in `NLPSentiment`. If you
skip it, those four pages try to start with the hub's Python and may fail to import.

### 4. Run the hub

```bash
./hub/run_hub.sh
```

The first run creates `hub/.venv` and installs `hub/requirements.txt`. The suite then
opens at **http://localhost:8600**. Pick a module from the sidebar. Embedded apps start
the first time you open their page, on ports 8601-8604. To stop the hub, press `Ctrl+C`
in the terminal.

### Data you have to supply

- **RAG Failure Diagnosis Agent:** the dataset `rag_failure_cases.csv` is not in the repo.
  Put it in `LearningAgent/` or `~/Downloads/`, or upload it on the module's page.
- **Transcode Time Predictor:** `kNNRegressorAgent/agent.py` reads the transcoding TSV from
  a hard-coded Downloads path. Point `DATA_PATH` at the copy in the repo,
  `KnnMLR/data/transcoding_mesurment.tsv`, before you open that page.
- **Corpus NLP Monitor:** uses the cached data in `CorpusNLP/data/` and works offline.
  The "Refresh data from the web" button downloads new data, which is slow.

### Troubleshooting

- **"The module did not start":** the page shows the last lines of the module's log. The
  full log is in `hub/logs/<module>.log`. Usually the module's `.venv` is missing (step 3).
- **Port already in use:** stop the other process, or run
  `./hub/run_hub.sh --server.port 8700`. Extra arguments are passed to Streamlit, and the
  embedded apps keep ports 8601-8604.
- **`No module named _tkinter`:** install Tk (see Requirements).

See [hub/README.md](hub/README.md) for how the modules are integrated.

Not in the repo: demo recordings (`*.mov`), project reports, generated outputs and logs the hub does not read, chatbot tests, virtual environments, and `.env`.

## Article Q&A Chatbot (`src/article_qa/`)

A source-grounded Q&A app. You give it one source and one question, and it answers only from
that text. It does a single turn per run and keeps no memory between runs.

- `ingest.py` loads article strings or PDF/text files into one grounded context
- `conversation.py` builds the prompt and handles the single-turn session
- `gemini_client.py` wraps the Gemini API
- `tts.py` adds optional spoken output and strips markdown before speaking
- `app.py` is the CLI entry point

```bash
export GEMINI_API_KEY="your-key"
PYTHONPATH=src python -m article_qa --text "Article text." --question "What is the main argument?"
PYTHONPATH=src python -m article_qa --files /path/to/article.pdf --question "Summarize the article."
PYTHONPATH=src python -m article_qa --text "Article text." --question "Summarize it." --speak
```

The prompt keeps answers inside the source text, in plain spoken English with no markdown.
If no TTS provider is configured, the app runs in text-only mode.
