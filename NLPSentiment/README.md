# Developer Sentiment Comparative Analyzer

**Research question:** Can off-the-shelf sentiment models understand how software developers talk?

Developer text is full of words that sound emotional but aren't ("kill the process", "fatal error",
"git commit", "return true"). This project compares three pre-trained sentiment models on two
hand-labeled developer datasets, shows where they fail, improves the lexicon model with
tech-specific tuning.

It extends the professor's `SentimentTextblob` class (binary, one model) to
**3 classes** (negative / neutral / positive) and **3 models + 1 tuned variant**,
keeping the same method names: `load_data → compute_sentiment → quantitative_sentiment →
visualize_sentiment → evaluation_confusion_matrix`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins a few versions because this was built on an **Intel Mac**, where the newest
PyTorch is 2.2.2 (so `numpy<2`, `pandas<3`, `pyarrow<21`, `transformers==4.49.0`). On an
Apple Silicon Mac or Linux these pins still work.

The first run downloads the RoBERTa model (~500 MB) from Hugging Face.

## How to run

```bash
python prepare_data.py          # once: download + convert the datasets
python sentiment_agent.py       # everything: score -> evaluate -> tuning -> outputs
streamlit run dashboard.py      # view the saved results
```

The first `sentiment_agent.py` run takes about 3 minutes (RoBERTa on CPU). Later runs take
about 5 seconds because RoBERTa predictions are cached in `data/roberta_cache.csv`.

Settings are constants at the top of `sentiment_agent.py`: `SAMPLE_SIZE` (default 1000 rows per
dataset, `None` = all), the TextBlob neutral band (±0.1) and the VADER thresholds (±0.05).

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads both datasets into `data/raw/`, converts each to `text,label` CSV |
| `sentiment_agent.py` | Main class `SentimentAgent` (professor's design) + evaluation, agreement, tuning, findings |
| `dashboard.py` | Streamlit dashboard over `outputs/` |

## Datasets (cite these)

| Dataset | Rows | neg / neu / pos | Source |
|---|---|---|---|
| Stack Overflow (Senti4SD) | 4,423 | 27% / 38% / 35% | Calefato, Lanubile, Maiorano, Novielli. *Sentiment Polarity Detection for Software Development.* Empirical Software Engineering, 2018. https://github.com/collab-uniba/Senti4SD (train + test partitions merged) |
| GitHub gold standard | 7,122 | 29% / 42% / 28% | Novielli, Calefato, Dongiovanni, Girardi, Lanubile. *Can We Use SE-specific Sentiment Analysis Tools in a Cross-Platform Setting?* MSR 2020. https://doi.org/10.6084/m9.figshare.11604597 |

## Outputs

| Path | What it is |
|---|---|
| `data/stackoverflow.csv`, `data/github.csv` | Cleaned datasets (`text,label`) |
| `data/roberta_cache.csv` | Cached RoBERTa predictions (text, label, score, seconds per text) |
| `outputs/results_summary.csv` | Accuracy, macro F1, % neutral called negative, runtime: per model per dataset |
| `outputs/model_comparison.png` | Bar chart of macro F1 for all models and datasets |
| `outputs/confusion_<dataset>.png` | Confusion matrix for each model |
| `outputs/scatter_<dataset>.png` | Professor's polarity vs subjectivity plot, colored by the 3 true labels |
| `outputs/predictions_<dataset>.csv` | Every row with every model's prediction and score |
| `outputs/agreement_summary.csv` | How often the models agree (all 3 and pairwise) |
| `outputs/disagreements_<dataset>.csv` | Example posts where the models disagree |
| `outputs/tuning_examples.csv` | VADER before vs after the tech lexicon on example sentences |
| `outputs/findings.md` | Plain-English findings summary |
| `outputs/run.log` | Log of the last run |

## Innovations: where to find them

Search the code for `# INNOVATION #`.

| # | Innovation | Where |
|---|---|---|
| 1 | Multi-model comparison (accuracy, macro F1, confusion matrix, summary table) | `sentiment_agent.py`: `SentimentAgent.compute_sentiment`, `evaluation_confusion_matrix`, `plot_model_comparison` |
| 2 | Model agreement analysis + disagreement examples | `sentiment_agent.py`: `SentimentAgent.agreement_analysis` |
| 3 | Tech-domain lexicon tuning + text cleanup, before vs after | `sentiment_agent.py`: `TECH_LEXICON`, `clean_dev_text`, `make_tuned_vader`, `tuning_before_after` |
| 4 | Streamlit dashboard | `dashboard.py` |
| 5 | Exports + logging (CSVs, PNGs, runtime per model, run log) | `sentiment_agent.py`: `setup_logging`, `main`, `write_findings` |

## Results (1,000-row sample per dataset)

| Model | Stack Overflow F1 | GitHub F1 | Runtime (1,000 posts) |
|---|---|---|---|
| TextBlob | 0.69 | 0.70 | < 1 s |
| VADER | 0.71 | 0.65 | < 1 s |
| VADER + tech lexicon | 0.73 | 0.68 | < 1 s |
| RoBERTa (twitter) | **0.78** | **0.76** | ~70-110 s on CPU |

## Design choices and limitations

- **No training.** All models are used off the shelf, so the train/test split of Senti4SD is merged
  and everything is used for evaluation.
- **TextBlob neutral band ±0.1.** The professor's binary version used a cutoff of 0; 3 classes need a band.
- **The tech lexicon was partly chosen by looking at VADER's errors on these same datasets**
  (e.g., "commit", "value", "true" pushing neutral posts to positive). That makes the measured gain
  somewhat optimistic; a fair test would pick words on one dataset and test on the other.
- **The RoBERTa model was trained on tweets**, not developer text, so it's still "off the shelf" for this domain.
