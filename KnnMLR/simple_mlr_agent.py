"""
Learning Agent using MultiLinear Regression (MLR)
Predicts how long a cloud video-transcoding job will take (utime, seconds)

Implemented as Fully-Autonomous Agent with Human-on-the-Loop
(the agent predicts batches on its own; the human monitors its warnings and types Q to stop it)
"""

import os
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import (CATEGORICAL, NUMERIC, RANDOM_STATE, TEST_SIZE, build_knn, clean_percept_v2,
                    load_data, log_csv)

CHECK_EVERY = 5          # seconds between checks of the environment
DISAGREE_LIMIT = 0.5     # warn if MLR and kNN differ by more than 50%
MIN_PREDICTION = 0.1     # seconds; a run time can't be negative


class LearningAgent_MLR:
    """ Fully-autonomous agent that predicts utime with MLR and kNN, and learns from feedback """

    def __init__(self):
        self.stop_event = threading.Event()      # set when the human types Q

    def load_data(self):
        # Percepts: measured transcoding jobs (umem is dropped - it is only known after the job runs)
        self.X, self.y = load_data()

        print('\n RAW DATA')
        print(self.X.head().to_string(index=True))
        print('\n Data Size = ', self.X.shape)
        print('Target (utime) Description')
        print(self.y.describe())

    def visualizeData_bef_modeling(self):
        print('\n VISUALIZE DATA PATTERNS BEFORE MODELING')
        self.plot_codec_boxplot()
        plt.show()

    def plot_codec_boxplot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        self.X.assign(utime=self.y).boxplot(column='utime', by='o_codec', showfliers=False, ax=ax)
        fig.suptitle('')
        ax.set(xlabel='Output codec', ylabel='utime (s)', title='Transcoding time by output codec')
        return fig

    def data_preprocessing(self):
        # Same split as the kNN assignment: 80/20, random_state=42
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Remember what "normal" looks like, for the warnings (improvement 3)
        self.ranges = self.X_train[NUMERIC].agg(['min', 'max'])
        self.known_codecs = set(self.X_train['codec']) | set(self.X_train['o_codec'])
        print(f'\n Train rows = {len(self.X_train)}, Test rows = {len(self.X_test)}')

    # ---------------- Learning Element ----------------
    def train_agents(self):
        # kNN = my tuned kNN agent (k=3, distance). MLR reuses the kNN's exact preprocessing step
        # (impute + scale numbers + one-hot codecs), so the comparison is fair.
        self.knn = build_knn()
        self.mlr = Pipeline([('prep', build_knn().named_steps['prep']),
                             ('mlr', LinearRegression())])
        self.mlr.fit(self.X_train, self.y_train)
        self.knn.fit(self.X_train, self.y_train)

        self.pred_mlr = np.clip(self.mlr.predict(self.X_test), MIN_PREDICTION, None)
        self.pred_knn = self.knn.predict(self.X_test)

    # ---------------- Critic ----------------
    def score_models(self):
        y_test = self.y_test
        results = {}
        for name, pred in [('MLR', self.pred_mlr), ('kNN', self.pred_knn)]:
            mse = mean_squared_error(y_test, pred)
            results[name] = {'MAE': mean_absolute_error(y_test, pred), 'MSE': mse,
                             'RMSE': np.sqrt(mse), 'R2': r2_score(y_test, pred)}
            log_csv('simple_evaluations.csv', {'model': name, 'test_rows': len(y_test), **results[name]})
        return results

    def share_far_off(self, pred):
        # Share of test predictions more than 50% away from the true utime
        return np.mean(np.abs(self.y_test - pred) / self.y_test > 0.5)

    def evaluate_agents(self):
        """ Compare the MLR and kNN models on the same test set, and plot their predictions and residuals """
        print('\n PERFORMANCE EVALUATIONS (same test set for both models)')
        results = self.score_models()
        print(pd.DataFrame(results).round(3))

        winner = 'kNN' if results['kNN']['RMSE'] < results['MLR']['RMSE'] else 'MLR'
        print(f'\n Better model on this test set: {winner}')

        off_mlr, off_knn = self.share_far_off(self.pred_mlr), self.share_far_off(self.pred_knn)
        print(f' Predictions more than 50% off: MLR {off_mlr:.1%}, kNN {off_knn:.1%}')

        self.plot_comparison(results)
        plt.show()

    def plot_comparison(self, results):
        # Improvement 1: graphs comparing both models
        y_test = self.y_test
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for col, (name, pred, color) in enumerate([('MLR', self.pred_mlr, 'tab:blue'),
                                                   ('kNN', self.pred_knn, 'tab:orange')]):
            ax = axes[0, col]
            ax.scatter(y_test, pred, s=4, alpha=0.3, color=color)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit Line')
            ax.set(xscale='log', yscale='log', xlabel='True utime (s)', ylabel='Predicted utime (s)',
                   title=f"{name}: True vs Predicted (R² = {results[name]['R2']:.3f})")
            ax.legend()

            ax = axes[1, col]
            residuals = y_test - pred
            ax.scatter(pred, residuals, s=4, alpha=0.3, color=color)
            ax.axhline(0, color='red', linestyle='--')
            ax.set(xscale='log', xlabel='Predicted utime (s)', ylabel='True - Predicted (s)',
                   title=f'{name}: Residuals')
        fig.tight_layout()
        return fig

    # ---------------- Problem Generator ----------------
    def worst_job_types(self, n=3):
        """ Returns the job types that the MLR predicts worst, on average"""
        test = pd.DataFrame({'job_type': self.X_test['codec'] + ' -> ' + self.X_test['o_codec'],
                             'MLR_error': np.abs(self.y_test - self.pred_mlr),
                             'kNN_error': np.abs(self.y_test - self.pred_knn)})
        return test.groupby('job_type')[['MLR_error', 'kNN_error']].mean().sort_values(
            'MLR_error', ascending=False).head(n)

    def suggest_next_problems(self):
        """ Suggests the next problems (job types) the agent should learn from, based on the MLR's worst predictions """
        print('\n PROBLEM GENERATOR: job types the MLR predicts worst')
        worst = self.worst_job_types()
        print(worst.round(2))
        for job_type in worst.index:
            print(f' Suggestion: run more {job_type} jobs to learn this conversion better.')

    # ---------------- Performance Element ----------------
    def check_warnings(self, job, mlr_raw, mlr, knn):
        """ Checks for noisy/missing inputs, values outside the training range, unseen codecs, and large MLR-vs-kNN disagreement """
        warnings = []
        for col in NUMERIC:
            low, high = self.ranges[col]
            if pd.notna(job[col]) and not low <= job[col] <= high:
                warnings.append(f'{col}={job[col]:g} is outside the training range [{low:g}, {high:g}]')
        for col in CATEGORICAL:
            if pd.notna(job[col]) and job[col] not in self.known_codecs:
                warnings.append(f"{col}='{job[col]}' was never seen in training")
        if mlr_raw < MIN_PREDICTION:
            warnings.append(f'MLR predicted {mlr_raw:.2f}s, clipped to {MIN_PREDICTION}s')
        disagreement = abs(mlr - knn) / max(min(mlr, knn), MIN_PREDICTION)
        if disagreement > DISAGREE_LIMIT:
            warnings.append(f'MLR and kNN disagree by {disagreement:.0%} - check this job')
        return warnings

    def predict_jobs(self, df_new):
        """ Predicts the utime for a batch of new jobs, and checks for warnings """
        cleaned = [clean_percept_v2(row.to_dict()) for _, row in df_new.iterrows()]
        X_new = pd.DataFrame([job for job, _ in cleaned], columns=NUMERIC + CATEGORICAL)

        mlr_raw = self.mlr.predict(X_new)
        mlr = np.clip(mlr_raw, MIN_PREDICTION, None)
        knn = self.knn.predict(X_new)
        X_new['MLR_pred'], X_new['kNN_pred'] = mlr.round(2), knn.round(2)

        job_warnings = []
        for i, (job, input_notes) in enumerate(cleaned):
            warnings = input_notes + self.check_warnings(job, mlr_raw[i], mlr[i], knn[i])
            # Improvement 2: log every prediction with its inputs and warnings
            log_csv('simple_predictions.csv', {**job, 'MLR_pred': mlr[i], 'kNN_pred': knn[i],
                                               'warnings': ' | '.join(warnings)})
            job_warnings.append(warnings)
        return X_new, job_warnings

    def predict_from_environment(self, batch_file):
        """ Checks for a new file, predicts the jobs in it, and learns from feedback if available """
        if not os.path.exists(batch_file):
            return False

        try:
            df_new = pd.read_csv(batch_file)
            if df_new.empty:
                return False

            print('\n New batch detected! Predicting...')
            X_new, job_warnings = self.predict_jobs(df_new)
        except Exception as e:
            # A bad file must not kill an autonomous agent: report it and set the file aside
            print(f"\n Could not process '{batch_file}': {e}")
            os.replace(batch_file, batch_file + '.error')
            return False

        print(X_new[['codec', 'o_codec', 'o_width', 'o_height', 'MLR_pred', 'kNN_pred']].to_string())
        for i, warnings in enumerate(job_warnings):
            for w in warnings:
                print(f'   WARNING job {i}: {w}')

        # Feedback: if the file says how long the jobs really took, learn from it
        if 'utime' in df_new.columns:
            self.learn_from_feedback(X_new[NUMERIC + CATEGORICAL], df_new['utime'])

        # Delete file after prediction (tells the environment it was consumed)
        os.remove(batch_file)
        print(f"\n Predictions done. File '{batch_file}' deleted.")
        return True

    # ---------------- Learning Element (feedback) ----------------
    def learn_from_feedback(self, X_done, y_done):
        """ Learns from the jobs that have finished and reported their utime. Retrains both models on the new data. """
        known = pd.to_numeric(y_done, errors='coerce').notna().to_numpy()
        if not known.any():
            return
        self.X_train = pd.concat([self.X_train, X_done[known]], ignore_index=True)
        self.y_train = pd.concat([self.y_train, pd.to_numeric(y_done[known])], ignore_index=True)
        self.mlr.fit(self.X_train, self.y_train)
        self.knn.fit(self.X_train, self.y_train)
        print(f'\n Learned from {known.sum()} finished job(s); both models retrained '
              f'on {len(self.X_train)} rows.')

    def listen_for_quit(self):
        # Human-on-the-loop: runs in the background and waits for Q
        while not self.stop_event.is_set():
            if input().strip().lower() == 'q':
                self.stop_event.set()

    def run_autonomously(self, batch_file):
        threading.Thread(target=self.listen_for_quit, daemon=True).start()
        print('\n Agent is running on its own. Type Q and press Enter to stop it.')

        try:
            while not self.stop_event.is_set():
                if not self.predict_from_environment(batch_file):
                    print('...Waiting for new batch file...')
                self.stop_event.wait(CHECK_EVERY)        # sleeps
        except KeyboardInterrupt:
            pass

        print('\n Exiting real-time prediction mode. Goodbye!')


# ---------------------------- Class Driver ----------------------------
if __name__ == '__main__':
    agent = LearningAgent_MLR()

    agent.load_data()
    agent.visualizeData_bef_modeling()
    agent.data_preprocessing()
    agent.train_agents()
    agent.evaluate_agents()
    agent.suggest_next_problems()

    # Agent designed as Fully-autonomous with files, and a human on the loop (Q to quit)
    batch_file = 'transcode_batch.csv'

    agent.run_autonomously(batch_file)
