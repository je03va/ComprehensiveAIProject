"""kNN Regressor Learning Agent: predicts video transcoding time (utime) for a cloud job scheduler."""
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = '/Users/je03va/Downloads/online+video+characteristics+and+transcoding+time+dataset/transcoding_mesurment.tsv'
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "agent.joblib"
LOG_DIR = BASE_DIR / "logs"
PLOT_DIR = BASE_DIR / "plots"

RANDOM_STATE = 42
TARGET = "utime"
LEAKAGE = ["id", "umem"]  # umem is only known after the job runs
CATEGORICAL = ["codec", "o_codec"]
NUMERIC = ["duration", "width", "height", "bitrate", "framerate", "i", "p", "b", "frames",
           "i_size", "p_size", "b_size", "size", "o_bitrate", "o_framerate", "o_width", "o_height"]
ALL_FEATURES = NUMERIC + CATEGORICAL
# frames = i + p + b and size = i_size + p_size + b_size; b_size is always 0
REDUNDANT = ["frames", "size", "b_size"]
FEATURE_SETS = {
    "all": ALL_FEATURES,
    "trimmed": [f for f in ALL_FEATURES if f not in REDUNDANT],
}
MUST_BE_POSITIVE = ["duration", "width", "height", "bitrate", "framerate",
                    "o_bitrate", "o_framerate", "o_width", "o_height"]

BASELINE_K = 5            # the "class example": fixed k, uniform weights, one split
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]
CV_FOLDS = 5
SPREAD_ALERT = 0.5        # neighbor std / neighbor mean above this -> low confidence
HIGH_ERROR_FACTOR = 3     # Critic flags |error| > factor * MAE


# ---------- Data ----------

def load_data(path=DATA_PATH, sample_size=None):
    """Load the TSV, drop leakage columns, optionally subsample for speed."""
    df = pd.read_csv(path, sep="\t").drop(columns=LEAKAGE)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_STATE)
    return df


def build_pipeline(features, k=BASELINE_K, weights="uniform"):
    """Preprocessing + kNN in one Pipeline, so scaling is fit on training data only."""
    numeric = [f for f in features if f in NUMERIC]
    categorical = [f for f in features if f in CATEGORICAL]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])  # columns not listed (e.g. trimmed ones) are dropped
    return Pipeline([("prep", preprocessor),
                     ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights))])


def clean_percept(raw):
    """Noisy percepts: turn raw user input into a job dict; bad values become NaN (imputed later)."""
    job, warnings = {}, []
    for col in NUMERIC:
        value = pd.to_numeric(raw.get(col), errors="coerce")
        if pd.isna(value) or not np.isfinite(value):
            warnings.append(f"{col}: missing or not a number ({raw.get(col)!r}), using training median.")
            value = np.nan
        elif value < 0 or (value == 0 and col in MUST_BE_POSITIVE):
            warnings.append(f"{col}: invalid value {value}, using training median.")
            value = np.nan
        job[col] = value
    for col in CATEGORICAL:
        value = str(raw.get(col) or "").strip().lower()
        if not value:
            warnings.append(f"{col}: missing, using most common codec.")
            value = np.nan
        job[col] = value
    return job, warnings


def log_csv(filename, row):
    """Append one timestamped row to a CSV log file."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / filename
    entry = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **row}])
    entry.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------- Learning Agent components ----------

class PerformanceElement:
    """Predicts utime for new jobs and reports how confident it is."""

    def __init__(self, model, X_train, y_train):
        """Store the fitted model, training utimes and ranges of numeric features for confidence checks."""
        self.model = model
        self.y_train = y_train.to_numpy()
        self.ranges = X_train[NUMERIC].agg(["min", "max"])
        self.known_codecs = {col: set(X_train[col]) for col in CATEGORICAL}

    def predict(self, X):
        return self.model.predict(X)

    def predict_job(self, job):
        X = pd.DataFrame([job], columns=ALL_FEATURES)
        prediction = float(self.predict(X)[0])

        # Look up the k nearest training jobs and how much their utimes disagree
        X_scaled = self.model.named_steps["prep"].transform(X)
        _, idx = self.model.named_steps["knn"].kneighbors(X_scaled)
        neighbor_times = self.y_train[idx[0]]
        spread = neighbor_times.std() / neighbor_times.mean()

        alerts = [] #collect warnings about the prediction's reliability, whether due to disagreement among neighbors or out-of-range values
        if spread > SPREAD_ALERT:
            alerts.append(f"Neighbors disagree: their utimes range {neighbor_times.min():.1f}-"
                          f"{neighbor_times.max():.1f}s (spread {spread:.2f} > {SPREAD_ALERT}).")
        for col in NUMERIC:
            low, high = self.ranges[col]
            if pd.notna(job[col]) and not low <= job[col] <= high:
                alerts.append(f"{col}={job[col]:g} is outside the training range [{low:g}, {high:g}].")
        for col in CATEGORICAL:
            if pd.notna(job[col]) and job[col] not in self.known_codecs[col]:
                alerts.append(f"{col}='{job[col]}' was never seen in training.")

        return {"prediction": prediction, "neighbor_times": neighbor_times, "spread": spread,
                "confidence": "Low" if alerts else "High", "alerts": alerts}


class LearningElement:
    """Selects feature set, k and distance weighting by cross-validation, then fits the model."""

    def __init__(self, k_values=K_VALUES, cv_folds=CV_FOLDS):
        """initialize into different k values and number of folds for cross-validation"""
        self.k_values = k_values
        self.cv = KFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)

    def search(self, X_train, y_train):
        rows = []
        for name, features in FEATURE_SETS.items():
            grid = GridSearchCV(
                build_pipeline(features),
                {"knn__n_neighbors": self.k_values, "knn__weights": ["uniform", "distance"]},
                cv=self.cv, scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error"},
                refit=False,
            )
            grid.fit(X_train, y_train)
            res = grid.cv_results_ #turning the cv_results_ into a DataFrame for easier manipulation
            for params, r2, rmse in zip(res["params"], res["mean_test_R2"], res["mean_test_RMSE"]):
                rows.append({"features": name, "k": params["knn__n_neighbors"],
                             "weights": params["knn__weights"], "R2": r2, "RMSE": -rmse})
        return pd.DataFrame(rows)

    def fit(self, X_train, y_train):
        """Finds the best feature set, k and distance weighting by cross-validation, then fits the model."""
        self.cv_results = self.search(X_train, y_train)
        best = self.cv_results.loc[self.cv_results["R2"].idxmax()]
        self.best_params = {"features": best["features"], "k": int(best["k"]), "weights": best["weights"]}
        model = build_pipeline(FEATURE_SETS[best["features"]], best["k"], best["weights"])
        return model.fit(X_train, y_train)


class Critic:
    """Scores predictions and flags the ones with unusually high error."""

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
                "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}

    def flag_high_error(self, y_true, y_pred):
        errors = np.abs(np.asarray(y_true) - y_pred)
        threshold = HIGH_ERROR_FACTOR * errors.mean()
        return errors > threshold, threshold


class ProblemGenerator:
    """Suggests job types (codec + resolution combos) that are rare in training or poorly predicted."""

    JOB_TYPE = ["codec", "o_codec", "resolution", "o_resolution"]

    @staticmethod
    def _job_types(X):
        """Return a DataFrame with only the columns that define a job type, plus resolution strings."""
        return X.assign(resolution=X["width"].astype(str) + "x" + X["height"].astype(str),
                        o_resolution=X["o_width"].astype(str) + "x" + X["o_height"].astype(str))

    def suggest(self, X_train, X_test, y_test, y_pred, top_n=10):
        """Return a DataFrame of the top_n job types that are rare in training or have high error in testing."""
        errors = np.abs(y_test.to_numpy() - y_pred)
        test = self._job_types(X_test).assign(abs_error=errors)
        stats = test.groupby(self.JOB_TYPE)["abs_error"].agg(test_mae="mean", test_jobs="size")
        stats["train_jobs"] = self._job_types(X_train).groupby(self.JOB_TYPE).size()
        stats["train_jobs"] = stats["train_jobs"].fillna(0).astype(int)

        rare = stats["train_jobs"] < stats["train_jobs"].quantile(0.10)
        hard = stats["test_mae"] > 2 * errors.mean()
        stats["reason"] = np.select([rare & hard, rare, hard],
                                    ["rare and high error", "few training samples", "high error"], "")
        return (stats[rare | hard].sort_values("test_mae", ascending=False)
                .head(top_n).reset_index())


class LearningAgent:
    """Wires the four components together: learn, act, criticise, explore."""

    def __init__(self):
        self.learner = LearningElement()
        self.critic = Critic()
        self.generator = ProblemGenerator()

    def train(self, df):
        """Split the data, fit the baseline and tuned models, evaluate both, flag high-error test jobs."""
        X, y = df.drop(columns=TARGET), df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE)

        baseline = build_pipeline(ALL_FEATURES).fit(X_train, y_train)
        self.baseline_metrics = self.critic.evaluate(y_test, baseline.predict(X_test))

        self.performer = PerformanceElement(self.learner.fit(X_train, y_train), X_train, y_train)
        self.y_test, self.y_pred = y_test, self.performer.predict(X_test)
        self.metrics = self.critic.evaluate(y_test, self.y_pred)
        self.high_error, self.error_threshold = self.critic.flag_high_error(y_test, self.y_pred)
        self.suggestions = self.generator.suggest(X_train, X_test, y_test, self.y_pred)

        log_csv("evaluations.csv", {"model": f"baseline (k={BASELINE_K}, uniform, all)",
                                    "rows": len(df), **self.baseline_metrics})
        log_csv("evaluations.csv", {"model": "tuned ({features}, k={k}, {weights})".format(**self.learner.best_params),
                                    "rows": len(df), **self.metrics})
        return self

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH):
        return joblib.load(path)


# ---------- Graphs ----------

def plot_cv_curves(cv_results):
    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(11, 4))
    for (features, weights), group in cv_results.groupby(["features", "weights"]):
        label = f"{features}, {weights}"
        ax_r2.plot(group["k"], group["R2"], marker="o", label=label)
        ax_rmse.plot(group["k"], group["RMSE"], marker="o", label=label)
    ax_r2.set(title="Cross-validated R² vs k", xlabel="k", ylabel="R²")
    ax_rmse.set(title="Cross-validated RMSE vs k", xlabel="k", ylabel="RMSE (s)")
    ax_r2.legend()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    limit = max(y_true.max(), y_pred.max())
    ax.plot([0, limit], [0, limit], color="red", linewidth=1, label="perfect prediction")
    ax.set(title="Actual vs predicted utime", xlabel="actual (s)", ylabel="predicted (s)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    residuals = np.asarray(y_true) - y_pred
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))
    ax_scatter.scatter(y_pred, residuals, s=4, alpha=0.3)
    ax_scatter.axhline(0, color="red", linewidth=1)
    ax_scatter.set(title="Residuals vs predicted", xlabel="predicted (s)", ylabel="actual - predicted (s)")
    ax_hist.hist(residuals, bins=100)
    ax_hist.set(title="Residual distribution", xlabel="actual - predicted (s)", ylabel="jobs")
    fig.tight_layout()
    return fig
