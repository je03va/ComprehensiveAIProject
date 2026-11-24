# Results Report: MLR Learning Agent vs kNN for Video-Transcoding Time Prediction

**Author:** Jalen Elliott · **Date:** 2026-10-28
**Attached figures:** `codec_boxplot.png`, `mlr_vs_knn.png`

> Context for the reader: this is a class project about AI agent design. I am preparing a live demo
> (code walkthrough + output interpretation). Every number below comes from actually running the code.

---

## 1. What the project is

A **fully autonomous learning agent** that predicts how long a cloud video-transcoding job will take
(`utime`, in seconds) using **Multilinear Regression (MLR)**. It is compared against my previously tuned
**kNN regressor** (k = 3, distance-weighted) from an earlier assignment.

- **Autonomy model:** Fully autonomous with a *human-on-the-loop*. The agent watches a folder for a
  batch file (`transcode_batch.csv`), predicts every job in it on its own, prints warnings, and deletes
  the file. The human monitors the warnings and types `Q` to stop it.
- **Interfaces:** a terminal agent (`simple_mlr_agent.py`) and a Streamlit GUI (`app.py`) with tabs
  Data / Critic / Problem Generator / Predict / Logs.
- **Environment simulator:** `make_batch.py` writes 5 real jobs + 1 deliberately noisy job + 1 job with
  an unseen codec. With `--finished`, it also includes the true run times so the agent can learn.

### Learning-agent components → code

| Component | Code | Role |
|---|---|---|
| Learning Element | `train_agents`, `learn_from_feedback` | Fits MLR and kNN; retrains both when finished jobs report their real `utime` |
| Critic | `score_models`, `evaluate_agents` | MAE / MSE / RMSE / R² on the same test set; comparison graphs |
| Problem Generator | `worst_job_types`, `suggest_next_problems` | Finds the conversion types MLR predicts worst; suggests collecting more of them |
| Performance Element | `predict_jobs`, `check_warnings`, `run_autonomously` | Predicts new batches, flags suspicious inputs/predictions |

### Improvements over the class example
1. Graphs comparing MLR vs kNN (true vs predicted, residuals).
2. Timestamped CSV logging of every evaluation and every prediction (`logs/`).
3. Warnings for noisy/missing inputs, out-of-training-range values, unseen codecs, negative MLR
   predictions, and MLR–kNN disagreement > 50%. Input cleaning also maps codec aliases
   (`H.264` → `h264`) and fixes `frames ≠ i+p+b`.

---

## 2. Data

- **68,784 transcoding jobs, 19 features** (input video properties + requested output settings:
  codec, o_codec, resolution, bitrate, framerate, frame counts/sizes, etc.).
- `id` and `umem` dropped. `umem` (memory used) is only known *after* the job runs → data leakage.
- **Split:** 80/20, `random_state=42` → 55,027 train / 13,757 test. Identical split **and identical
  preprocessing** (impute, scale numerics, one-hot codecs) for both models, so only the algorithm differs.

**Target `utime` is heavily right-skewed:**

| min | 25% | median | mean | 75% | max |
|---|---|---|---|---|---|
| 0.18 s | 2.10 s | 4.41 s | 10.0 s | 10.4 s | 224.6 s |

### Figure: `codec_boxplot.png`: utime by output codec (outliers hidden)

| Output codec | Median | Middle 50% (IQR) | Max |
|---|---|---|---|
| flv | 1.5 s | 0.9 – 3.6 s | 13 s |
| mpeg4 | 3.9 s | 2.4 – 7.2 s | 21 s |
| vp8 | 6.3 s | 3.0 – 15.4 s | 109 s |
| **h264** | **11.1 s** | **4.9 – 26.9 s** | **225 s** |

**Interpretation:** Output codec has a large effect on run time (h264 ≈ 7× slower than flv at the median).
The slow codecs also have the most spread, which means their cost depends on other features
(resolution, duration) *multiplicatively*. That is an interaction a linear model cannot capture.

---

## 3. Critic results (same 13,757-job test set)

| Metric | MLR | kNN | Better |
|---|---|---|---|
| MAE (s) | 5.738 | **1.104** | kNN ≈ 5.2× lower |
| MSE | 111.747 | **10.647** | |
| RMSE (s) | 10.571 | **3.263** | kNN ≈ 3.2× lower |
| R² | 0.563 | **0.958** | |
| Predictions > 50% off | 66.4% | **2.4%** | |

**Winner: kNN** on every metric.

### Figure: `mlr_vs_knn.png` (2×2: true-vs-predicted on log axes, and residuals)

- **MLR, true vs predicted:** a wide cloud. It **overpredicts short jobs** (the bulk of points sit above
  the diagonal below ~10 s) and **cannot predict long jobs**: predictions level off around 40–50 s
  (max MLR prediction ≈ 52 s) even when the true time is 100–200 s.
- **MLR floor:** a flat line of points at 0.1 s. **24% of MLR's raw test predictions were negative**
  and got clipped to the 0.1 s minimum.
- **MLR residuals:** a curved, funnel-shaped pattern. Negative (overprediction) in the mid range, then
  rising sharply to +100 s or more at the top. A curve in the residuals is the classic sign that the
  true relationship is non-linear.
- **kNN:** points hug the diagonal across the whole range. Residuals stay centered on 0, widening only
  for the longest jobs (> 30 s), where there are fewer similar neighbours.

### Why MLR loses
1. **Non-linearity / interactions.** Run time ≈ (frames × pixels × codec cost). That is multiplicative,
   but MLR can only add weighted features.
2. **Skewed target.** One straight line can't fit both the many 1–5 s jobs and the rare 100+ s jobs.
3. **Multicollinearity.** The largest MLR coefficients are `size` (−96.9) and `p_size` (+91.4).
   These two features are nearly the same quantity, so the model makes them cancel each other out.
   The individual coefficients are not interpretable.
4. **kNN suits this data.** The dataset contains the same source videos transcoded to many output
   settings, so very close neighbours almost always exist.

Other notable MLR coefficients (standardised features): `o_codec=h264` +11.2 s, `o_width` +9.8,
`o_codec=flv` −7.5, `o_codec=mpeg4` −4.8. Intercept ≈ 9.9 s. These agree with the boxplot.

---

## 4. Problem Generator results (mean absolute error per conversion type)

| Job type (input → output) | MLR error (s) | kNN error (s) |
|---|---|---|
| **h264 → h264** | **12.21** | 3.52 |
| **flv → h264** | **12.08** | 2.93 |
| **vp8 → h264** | **12.01** | 2.76 |
| mpeg4 → h264 | 9.35 | 2.23 |
| vp8 → flv | 4.42 | 0.13 |
| h264 → flv | 4.38 | 0.18 |
| vp8 → vp8 | 4.07 | 0.89 |
| vp8 → mpeg4 | 4.00 | 0.33 |
| h264 → mpeg4 | 3.96 | 0.35 |
| h264 → vp8 | 3.56 | 1.06 |
| flv → mpeg4 | 3.50 | 0.26 |
| flv → flv | 3.45 | 0.08 |
| mpeg4 → flv | 3.35 | 0.07 |
| mpeg4 → vp8 | 3.11 | 0.74 |
| flv → vp8 | 3.08 | 0.98 |
| mpeg4 → mpeg4 | 2.68 | 0.21 |

**Agent output:** "Suggestion: run more h264 → h264 / flv → h264 / vp8 → h264 jobs to learn this
conversion better."

**Interpretation:**
- The **output** codec drives error; the input codec barely matters. All 4 worst types output h264,
  then error drops sharply from about 9–12 s to about 3–4 s.
- kNN is better on every type. The relative gap is largest on cheap codecs (flv output: kNN ≈ 0.1 s
  vs MLR ≈ 4 s, on jobs with a median of 1.5 s).
- **Honest caveat:** h264 is already ~25% of the data (17k jobs), so "collect more" will not fix MLR.
  The real fix is model form: interaction features (e.g. pixels × frames × codec), a log-transformed
  target, or a non-linear model. The kNN comparison is evidence for this.
- In the GUI tab: a slider (3–15) sets how many types are shown, with a horizontal bar chart (MLR vs kNN
  per type), the table, and suggestions for the top 3. *Known limitation:* this tab uses the initial
  test predictions and does not refresh after feedback retraining.

---

## 5. Performance Element: example batch run

Batch from `make_batch.py --finished` (5 real jobs with known times + 2 problem jobs):

| # | codec → o_codec | Output res | MLR pred | kNN pred | True utime | Warnings |
|---|---|---|---|---|---|---|
| 0 | vp8 → mpeg4 | 640×480 | 1.62 | 4.30 | 4.30 | MLR/kNN disagree 166% |
| 1 | h264 → mpeg4 | 1920×1080 | 17.90 | 8.64 | 8.64 | disagree 107% |
| 2 | h264 → flv | 480×360 | 0.10 | 1.37 | 1.46 | MLR predicted −6.47 s, clipped; disagree 1271% |
| 3 | mpeg4 → h264 | 1920×1080 | 33.86 | 58.81 | 58.81 | disagree 74% |
| 4 | vp8 → h264 | 1280×720 | 27.53 | 33.16 | 33.16 | — |
| 5 | vp8 → mpeg4 (noisy) | **9999**×480 | 151.32 | 15.44 | — | `duration='abc'` → training median used; `o_width=9999` outside [176, 1920]; disagree 880% |
| 6 | **av1** → mpeg4 | 1920×1080 | 17.12 | 8.42 | — | codec 'av1' never seen in training; disagree 103% |

Then: *"Learned from 5 finished job(s); both models retrained on 55,032 rows."* (55,027 + 5).

**Interpretation:**
- The warnings work as designed. Bad input is repaired and flagged, never crashes the agent.
  Out-of-range and unseen values are flagged, and the human is told which jobs to check.
- Job 5 shows the extrapolation risk: MLR extends its straight line to a 9999-px width and predicts
  151 s, while kNN stays bounded by its neighbours (15 s).
- Job 2 shows MLR predicting a **negative** time, which is physically impossible. That justifies the
  clipping safeguard.
- **Caveat:** kNN matches the true times almost exactly here because `make_batch.py` samples from the
  full dataset, so these jobs may already be in kNN's training set (k=3, distance weighting returns a
  stored point exactly). This demo batch shows the *agent mechanics*; the honest accuracy numbers are
  the test-set results in Section 3.

---

## 6. Conclusions

1. **kNN is clearly the better predictor** for this problem: R² 0.96 vs 0.56, MAE 1.1 s vs 5.7 s, on
   an identical split and identical preprocessing.
2. **MLR captures the right trends** (h264 output and larger resolution make jobs longer), and its
   coefficients agree with the exploratory boxplot. But a linear form cannot represent the
   multiplicative cost of transcoding. Evidence: 24% negative predictions, a ~50 s prediction ceiling,
   and curved residuals.
3. **Running both models is still useful.** Large MLR–kNN disagreement is a cheap uncertainty signal
   that the human-on-the-loop can act on.
4. **Next steps:** log-transform `utime`; add interaction features (output pixels × frames, codec ×
   resolution); drop redundant size columns; retest MLR. Fix the Problem Generator GUI tab so it
   refreshes after feedback retraining. Build demo batches from the test set only, so batch accuracy
   isn't inflated.
