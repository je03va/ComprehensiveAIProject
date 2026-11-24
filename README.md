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

## Run the suite

```bash
cp .env.example .env        # add GEMINI_API_KEY for the chatbot
./hub/run_hub.sh            # opens http://localhost:8600
```

The first run creates `hub/.venv` from `hub/requirements.txt`. Four modules
(kNN regressor, MLR, Corpus NLP, Sentiment) run as separate Streamlit apps embedded in
the hub. Each one uses its own `.venv` because they need different NumPy versions.
Create those once:

```bash
for d in kNNRegressorAgent KnnMLR CorpusNLP NLPSentiment; do
  python3 -m venv $d/.venv && $d/.venv/bin/pip install -r $d/requirements.txt
done
```

If a module has no `.venv`, the hub falls back to its own Python. See
[hub/README.md](hub/README.md) for how the modules are integrated, plus data notes.

Not in the repo: demo recordings (`*.mov`), virtual environments, `.env`, and hub logs.

## Article Q&A Chatbot (`src/article_qa/`)

A source-grounded Q&A app. You give it one source and one question, and it answers only from
that text. It does a single turn per run and keeps no memory between runs.

- `ingest.py` loads article strings or PDF/text files into one grounded context
- `conversation.py` builds the prompt and handles the single-turn session
- `gemini_client.py` wraps the Gemini API
- `tts.py` adds optional spoken output and strips markdown before speaking
- `app.py` is the CLI entry point; tests are in `tests/test_article_qa.py`

```bash
export GEMINI_API_KEY="your-key"
PYTHONPATH=src python -m article_qa --text "Article text." --question "What is the main argument?"
PYTHONPATH=src python -m article_qa --files /path/to/article.pdf --question "Summarize the article."
PYTHONPATH=src python -m article_qa --text "Article text." --question "Summarize it." --speak
```

The prompt keeps answers inside the source text, in plain spoken English with no markdown.
If no TTS provider is configured, the app runs in text-only mode.
