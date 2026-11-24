# Project Report: Developer Sentiment Comparative Analyzer

> Context for the reader (Claude): this is a complete summary of a student NLP project, written so you can
> help with write-ups, slides, analysis, or follow-up questions without access to the code. Every number
> below comes from the project's last run (`outputs/run.log`, 2026-11-08). Nothing has been estimated.

## 1. Research question

**Can off-the-shelf sentiment models understand how software developers talk?**

Developer text is full of words that sound emotional but are technical ("kill the process", "fatal error",
"git commit", "return true", "free memory"). General-purpose sentiment models may misread them. The project
compares pre-trained models on two hand-labeled developer datasets, measures where they fail, and tries to
fix the lexicon model with tech-specific tuning.

It extends a course-provided `SentimentTextblob` class (1 model, binary labels) to **3 classes**
(negative / neutral / positive) and **4 models**, keeping the same method pipeline:
`load_data → compute_sentiment → quantitative_sentiment → visualize_sentiment → evaluation_confusion_matrix`.

## 2. Data

| Dataset | Full size | neg / neu / pos (full) | Source |
|---|---|---|---|
| Stack Overflow (Senti4SD) | 4,423 | 27% / 38% / 35% | Calefato, Lanubile, Maiorano, Novielli. *Sentiment Polarity Detection for Software Development.* Empirical Software Engineering, 2018. Train (3,098) + test (1,326) partitions merged. |
| GitHub gold standard | 7,122 | 29% / 42% / 28% | Novielli, Calefato, Dongiovanni, Girardi, Lanubile. *Can We Use SE-specific Sentiment Analysis Tools in a Cross-Platform Setting?* MSR 2020. doi:10.6084/m9.figshare.11604597 |

**Evaluation sample:** 1,000 random rows per dataset (seed 42).

| Sample | negative | neutral | positive | median length |
|---|---|---|---|---|
| Stack Overflow | 250 | 386 | 364 | 114 chars |
| GitHub | 291 | 406 | 303 | 90.5 chars |

Cleaning: labels lower-cased and restricted to the 3 classes, empty texts dropped, `\r\n` → `\n`,
stray trailing `"` stripped from GitHub texts.

## 3. Models

| Model | How it labels | Thresholds |
|---|---|---|
| `textblob` | TextBlob polarity (-1..1) | > +0.1 positive, < -0.1 negative, else neutral |
| `vader` | VADER compound score | standard ±0.05 |
| `vader_tuned` | VADER + custom tech lexicon + text cleanup (removes code blocks, inline code, URLs) | ±0.05 |
| `roberta` | Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`, argmax label (trained on tweets) | n/a |

No model was trained or fine-tuned. Everything is off the shelf except the hand-written lexicon.

**Tech lexicon (VADER -4..+4 scale), about 75 entries:**
- Set to neutral (0), "scary" technical words: kill/killed, fatal, exploit, attack, dead, deadlock, crash,
  error(s), warning, block(ed), reject(ed), destroy(ed), panic, dump, violation, threat, vulnerable/vulnerability,
  risk, critical, severe, defect, forced, lazy, argument(s), empty, missing.
- Set to neutral (0), "happy" technical words: commit(ted), value(s), true, solution(s), create(d), security,
  secure, support(ed), free(d), clean, original.
- Mildly negative (-0.5): fail/failed/fails/failing/failure, problem(s), wontfix.
- Developer slang: lgtm +2.0, hacky -1.0, kludge -1.0.

## 4. Headline results

| Model | SO accuracy | SO macro F1 | GitHub accuracy | GitHub macro F1 | Runtime (1,000 posts, CPU) |
|---|---|---|---|---|---|
| TextBlob | 0.693 | 0.685 | 0.700 | 0.704 | 0.2–0.35 s |
| VADER | 0.725 | 0.714 | 0.652 | 0.647 | ~0.1 s |
| VADER + tech lexicon | 0.736 | 0.726 | 0.684 | 0.683 | ~0.1 s |
| **RoBERTa** | **0.778** | **0.779** | **0.757** | **0.764** | 107 s (SO), 68 s (GitHub) |

- Best on both datasets: **RoBERTa** (mean macro F1 0.77).
- Worst: TextBlob on Stack Overflow (0.685), VADER on GitHub (0.647).
- RoBERTa is roughly 300–1,000× slower than the lexicon models for about +0.05 to +0.12 macro F1.

### "Neutral post called negative" rate (the dev-jargon failure)

| Model | Stack Overflow | GitHub |
|---|---|---|
| TextBlob | 10.6% | 8.1% |
| VADER | 20.7% | 22.9% |
| VADER + tech lexicon | 19.7% | 17.5% |
| RoBERTa | 18.4% | **26.1%** |

## 5. Per-class metrics (precision / recall / F1)

**Stack Overflow**

| Model | negative | neutral | positive |
|---|---|---|---|
| TextBlob | 0.75 / 0.58 / 0.65 | 0.66 / 0.60 / 0.63 | 0.69 / 0.87 / 0.77 |
| VADER | 0.69 / 0.79 / 0.74 | 0.84 / 0.46 / 0.60 | 0.69 / 0.96 / 0.80 |
| VADER tuned | 0.71 / 0.78 / 0.74 | 0.85 / 0.49 / 0.62 | 0.70 / 0.96 / 0.81 |
| RoBERTa | 0.72 / 0.81 / 0.77 | 0.73 / 0.72 / 0.72 | 0.88 / 0.82 / 0.85 |

**GitHub**

| Model | negative | neutral | positive |
|---|---|---|---|
| TextBlob | 0.83 / 0.64 / 0.72 | 0.69 / 0.63 / 0.66 | 0.64 / 0.85 / 0.73 |
| VADER | 0.65 / 0.71 / 0.68 | 0.85 / 0.42 / 0.56 | 0.57 / 0.91 / 0.70 |
| VADER tuned | 0.70 / 0.72 / 0.71 | 0.86 / 0.50 / 0.63 | 0.58 / 0.90 / 0.71 |
| RoBERTa | 0.67 / 0.85 / 0.75 | 0.73 / 0.70 / 0.72 | 0.93 / 0.74 / 0.83 |

## 6. Confusion matrices (rows = true label, columns = predicted; order neg, neu, pos)

**Stack Overflow (true counts: 250 / 386 / 364)**

| Model | true neg → | true neu → | true pos → |
|---|---|---|---|
| TextBlob | 145 / 78 / 27 | 41 / 232 / 113 | 7 / 41 / 316 |
| VADER | 198 / 26 / 26 | 80 / 179 / 127 | 9 / 7 / 348 |
| VADER tuned | 196 / 26 / 28 | 76 / 191 / 119 | 6 / 9 / 349 |
| RoBERTa | 203 / 45 / 2 | 71 / 277 / 38 | 6 / 60 / 298 |

**GitHub (true counts: 291 / 406 / 303)**

| Model | true neg → | true neu → | true pos → |
|---|---|---|---|
| TextBlob | 187 / 74 / 30 | 33 / 255 / 118 | 5 / 40 / 258 |
| VADER | 207 / 19 / 65 | 93 / 169 / 144 | 17 / 10 / 276 |
| VADER tuned | 209 / 21 / 61 | 71 / 203 / 132 | 18 / 13 / 272 |
| RoBERTa | 247 / 42 / 2 | 106 / 285 / 15 | 17 / 61 / 225 |

## 7. Tech-lexicon tuning: before vs after

**On the full samples (VADER → VADER tuned):**
- Stack Overflow: macro F1 0.714 → 0.726 (+0.012); neutral-called-negative 20.7% → 19.7%.
- GitHub: macro F1 0.647 → 0.683 (+0.036); neutral-called-negative 22.9% → 17.5%.

**On 8 hand-picked developer sentences** (tuning changed 6 of 8 labels):

| Sentence | VADER before | label | VADER after | label | RoBERTa |
|---|---|---|---|---|---|
| Kill the process and restart the server. | -0.691 | negative | 0.000 | neutral | negative |
| This bug is nasty, the app crashes on startup. | -0.557 | negative | -0.557 | negative | negative |
| Fatal error: cannot allocate memory. | -0.735 | negative | 0.000 | neutral | negative |
| The exploit was patched in the latest release. | -0.103 | negative | 0.000 | neutral | neutral |
| This method is deprecated, use the new API instead. | 0.000 | neutral | 0.000 | neutral | negative |
| Watch out for a deadlock when two threads block on the same lock. | -0.649 | negative | 0.000 | neutral | neutral |
| The attack surface is much smaller now. | -0.477 | negative | 0.000 | neutral | neutral |
| LGTM, ship it! | 0.000 | neutral | 0.616 | positive | positive |

Plain VADER calls 6 of 8 negative; RoBERTa calls 4 of 8 negative, including "Kill the process…" and
"Fatal error…". So the transformer is not immune to dev jargon either.

## 8. Model agreement (TextBlob, VADER, RoBERTa)

| Dataset | All 3 agree | Accuracy when all 3 agree | TextBlob–VADER | TextBlob–RoBERTa | VADER–RoBERTa |
|---|---|---|---|---|---|
| Stack Overflow | 57.3% | 90.9% | 69.6% | 69.9% | 70.8% |
| GitHub | 47.9% | 93.9% | 63.5% | 61.0% | 64.5% |

Agreement works as a confidence signal: when all three models agree, they are right 91–94% of the time,
well above any single model (65–78%).

**Example three-way disagreements** (true label → TextBlob / VADER / RoBERTa):
- SO: "StyleReports is one of the most heinous toolsuites I've ever had to use." true **negative** → positive / neutral / negative
- SO: "The question is not when, but what is the easiest and least painful method." true **neutral** → negative / positive / neutral
- SO: "…if someone wanted to, they could exploit your code anyway clientside…" true **positive** → positive / negative / neutral
- GitHub: "Ok, so let's be paranoid." true **negative** → positive / negative / neutral
- GitHub: "If you multiply-include `gst.h`, wouldn't it be good to add some include guards…" true **neutral** → positive / negative / neutral
- GitHub: "Umm, there's plenty in there that don't :P" true **positive** → positive / negative / neutral

## 9. Observations worth discussing

These come straight from the numbers above. Some go beyond the project's own auto-generated findings.

1. **RoBERTa wins overall, but by a modest margin.** +0.05 to +0.12 macro F1 over the lexicon models, at
   roughly 300–1,000× the runtime, using a model trained on tweets rather than developer text.
2. **VADER's biggest error is neutral → positive, not neutral → negative.** Of the truly neutral posts, VADER
   calls 127 (SO) and 144 (GitHub) positive, compared with 80 and 93 negative. Its neutral recall is only
   0.42–0.46. VADER is "too emotional" in both directions.
3. **RoBERTa has the highest neutral→negative rate on GitHub (26.1%)**, higher than plain VADER. That partly
   undercuts the "transformers understand dev jargon" story. RoBERTa does get neutral recall right (~0.70),
   but when it leaves neutral it leans negative.
4. **TextBlob is the most conservative about negativity** (lowest neutral→negative, 8–11%), but it misses many
   real negatives (negative recall 0.58–0.64).
5. **Lexicon tuning helps more on GitHub (+0.036 F1) than on Stack Overflow (+0.012).** On SO the gain is
   about the size of sampling noise for n=1,000.
6. **RoBERTa almost never confuses positive and negative** (2 true-negatives called positive on each dataset).
   Its errors are nearly all at the neutral boundary.
7. **GitHub is harder for agreement** (48% vs 57% all-agree). Its posts are shorter and use more emoticons and
   informal tone (":P").

## 10. Limitations (as stated by the project author)

- **No training.** All models are off the shelf, so Senti4SD train and test splits were merged and everything
  was used for evaluation.
- **Lexicon chosen partly by looking at VADER's errors on these same datasets** (e.g., "commit", "value",
  "true" pushing neutral posts to positive). The measured tuning gain is therefore optimistic. A fair test
  would build the lexicon on one dataset and evaluate on the other.
- **The RoBERTa model was trained on tweets**, not developer text.
- **1,000-row samples**, not the full datasets (configurable; `SAMPLE_SIZE = None` runs everything).
- **Thresholds were not tuned** (TextBlob ±0.1, VADER ±0.05).
- Additional note: the code comment above the 8 example sentences says they are "negative, but VADER reads as
  neutral". In fact most are *neutral* technical sentences that VADER reads as *negative*. The comment is wrong;
  the analysis is not affected.

## 11. Project deliverables ("innovations" beyond the base assignment)

1. Multi-model comparison: accuracy, macro F1, confusion matrices, summary table.
2. Model agreement analysis plus saved disagreement examples.
3. Tech-domain lexicon and text cleanup, with before/after comparison.
4. Streamlit dashboard (`dashboard.py`) over the saved outputs.
5. Exports and logging: CSVs, PNG charts, per-model runtime, run log; RoBERTa predictions cached so re-runs
   take about 5 s instead of about 3 min.

**Output files** (images can be attached separately if useful): `model_comparison.png` (macro F1 bar chart),
`confusion_stackoverflow.png`, `confusion_github.png`, `scatter_stackoverflow.png`, `scatter_github.png`
(TextBlob polarity vs subjectivity, colored by true label), plus `results_summary.csv`, `agreement_summary.csv`,
`predictions_<dataset>.csv` (every row and every model's prediction and score), `disagreements_<dataset>.csv`,
`tuning_examples.csv`, `findings.md`, `run.log`.

**Environment:** Python, built on an Intel Mac (PyTorch 2.2.2, `numpy<2`, `pandas<3`,
`transformers==4.49.0`). Libraries: pandas, scikit-learn, matplotlib, TextBlob, vaderSentiment, transformers,
Streamlit.
