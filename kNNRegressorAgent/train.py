"""Train the Learning Agent, print results, save graphs and the model."""
import argparse

import pandas as pd

from agent import (PLOT_DIR, LearningAgent, load_data, plot_actual_vs_predicted,
                   plot_cv_curves, plot_residuals)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, help="subsample this many rows for faster training")
args = parser.parse_args()

df = load_data(sample_size=args.sample)
print(f"Loaded {len(df)} rows, {df.shape[1] - 1} predictors")

agent = LearningAgent().train(df)

print("\nBest from cross-validation:", agent.learner.best_params)
best_per_set = agent.learner.cv_results.sort_values("R2").groupby("features").tail(1)
print("\nBest config per feature set (CV):\n", best_per_set.to_string(index=False))

comparison = pd.DataFrame({"baseline": agent.baseline_metrics, "tuned": agent.metrics})
print("\nTest-set metrics:\n", comparison.round(3))

print(f"\nCritic flagged {agent.high_error.sum()} / {len(agent.high_error)} test jobs "
      f"with |error| > {agent.error_threshold:.2f}s")
print("\nProblem Generator suggestions:\n", agent.suggestions.round(2).to_string(index=False))

PLOT_DIR.mkdir(exist_ok=True)
plot_cv_curves(agent.learner.cv_results).savefig(PLOT_DIR / "cv_curves.png")
plot_actual_vs_predicted(agent.y_test, agent.y_pred).savefig(PLOT_DIR / "actual_vs_predicted.png")
plot_residuals(agent.y_test, agent.y_pred).savefig(PLOT_DIR / "residuals.png")

agent.save()
print(f"\nSaved graphs to {PLOT_DIR}/ and model to models/")
