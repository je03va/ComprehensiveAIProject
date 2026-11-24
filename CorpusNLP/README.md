# Corpus Analysis NLP Agent: AI Coding Assistants

This agent tracks how public discussion and sentiment about AI coding assistants
(Copilot, ChatGPT/Codex, Claude, Cursor) change month by month. It also flags
emerging topics and risk indicators (security, jobs, bugs/hallucinations, licensing).
It builds on the professor's `Electronic_Books` starter class.

## Data sources (6, every document has a date)

| # | Source | Endpoint | Why |
|---|--------|----------|-----|
| 1 | Hacker News comments | `hn.algolia.com/api/v1/search` (per month, 5 queries) | Developer opinions |
| 2 | Stack Overflow questions | `api.stackexchange.com/2.3/search/advanced` | Practical problems users hit |
| 3 | GitHub issues | `api.github.com/search/issues` | Bug reports and feature requests |
| 4 | arXiv abstracts | `export.arxiv.org/api/query` ("code generation" + "large language model") | Research view |
| 5 | Tech news RSS | The Verge AI, TechCrunch AI, Ars Technica AI feeds | Media coverage (recent weeks only) |
| 6 | Project Gutenberg *R.U.R.* (1920) | `gutenberg.org/cache/epub/59112` | Historical baseline about machines replacing workers |

Sources 1 to 4 cover the whole window (Nov 2025 to Oct 2026) and drive the trends over
time. Sources 5 and 6 are snapshot/baseline comparisons. No API keys are needed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python agent.py                      # full analysis using cached data in data/
python agent.py --refresh            # re-download every source (about 3 minutes)
streamlit run app.py                 # interactive dashboard

# Repeatable evaluation: rerun on other time windows or sources
python agent.py --start 2025-11 --end 2026-04 --label h1
python agent.py --start 2026-05 --end 2026-10 --label h2
python agent.py --sources hn,stackoverflow,github --label developers
```

Fetched data is cached in `data/<source>.json`, so later runs work offline and give the
same results. If a download fails, the agent prints a message and uses the cached file.

## Outputs (`outputs/`, one set per `--label`)

- `findings_<label>.txt`: all tables, plus every alert with an example text and its URL
- `sentiment_over_time_*.png`, `sentiment_by_source_*.png`, `risk_trends_*.png`,
  `assistant_mentions_*.png`, `top_keywords_*.png`, `vader_vs_textblob_*.png`
- `documents_<label>.csv`: every analyzed document with its scores and risk flags
- `evaluation_<label>.csv`: VADER vs TextBlob comparison per source
- `alerts_log.csv` and `run_log.csv`: appended on every run, with timestamps

## Settings

Thresholds are plain constants at the top of `agent.py`: `SENTIMENT_DROP_THRESHOLD`,
`RISK_SPIKE_PCT`, `MIN_RISK_DOCS`, `MIN_KEYWORD_DOCS`, `RISK_KEYWORDS`, `START_MONTH`/`END_MONTH`.

## Innovations

1. Streamlit dashboard: `app.py`
2. Text cleaning pipeline: `clean_text()` in `agent.py`
3. Sentiment over time with VADER: `add_sentiment()`, `sentiment_over_time()`
4. Emerging topics and risk keywords: `add_flags()`, `rate_by_period()`, `emerging_topics()`
5. Automated alerts: `find_alerts()`
6. CSV logging: `log_results()`
7. Repeatable evaluation, VADER vs TextBlob: `compare_sentiment_methods()` and the `--sources/--start/--end/--label` options
