# MLR Learning Agent: cloud video-transcoding run-time prediction

A fully autonomous learning agent that predicts how long a transcoding job will take (`utime`, seconds)
with Multilinear Regression, and compares it to my tuned kNN agent (`../kNNRegressorAgent`, k=3, distance).
Human-on-the-loop: the agent predicts batches on its own; the human watches its warnings and types `Q` to stop it.

## Run

```bash
cd KnnMLR
source .venv/bin/activate          # first time: python -m venv .venv && pip install -r requirements.txt

python simple_mlr_agent.py         # terminal 1: train, evaluate, graphs, then wait for batches
python make_batch.py               # terminal 2: drop a batch of new jobs for the agent
python make_batch.py --finished    # same, with real run times -> the agent retrains (feedback)

streamlit run app.py               # or: the same agent as a browser GUI (tabs per component)
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI: data, Critic metrics/graphs, Problem Generator, predict/feedback, logs |
| `simple_mlr_agent.py` | The agent (Learning Element, Performance Element, Critic, Problem Generator) |
| `make_batch.py` | Simulates the environment: writes `transcode_batch.csv` |
| `common.py` | Reuses the kNN assignment's data loading, input cleaning and kNN pipeline |
| `data/` | Link to `transcoding_mesurment.tsv` |
| `logs/` | `simple_predictions.csv` and `simple_evaluations.csv` (created on first run) |

## Improvements over the class example

1. Graphs comparing MLR vs kNN (actual vs predicted, residuals)
2. File logging of predictions and evaluations with timestamps
3. Warnings for noisy/missing inputs, out-of-range values, unseen codecs and large MLR-kNN disagreement
