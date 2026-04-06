# ==========================================
# TEST FOR NONLINEAR INTERACTIONS
# Linear Regression vs XGBoost (20 runs total)
# ==========================================
# Goal:
#   Check whether the relationship between species abundances at time = 0
#   and the final "production output" is mostly linear or has strong nonlinear
#   interactions.
#
# How we do it:
#   - Run 20 independent experiments (5 different train/test splits × 4 random seeds)
#   - Train a simple Linear Regression and an XGBoost model on each
#   - Compute R² on train and test sets for both models
#   - Show mean R² ± 95% confidence interval
#   - Create nice plots (bar chart, scatter plots, box plots) so we can visually
#     compare the two models
#
# If XGBoost performs noticeably better than Linear Regression, it suggests
# there are important nonlinear relationships in the data.
# ==========================================

import warnings
warnings.filterwarnings("ignore")   # Clean up console output (XGBoost warnings)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# ==========================================
# CONFIGURATION (everything you might want to change)
# ==========================================
ABUNDANCE_CSV = r"C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/simulated_abundance_switch.csv"
OUTPUT_CSV    = r"C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/production_output_summary.csv"

TEST_SIZE = 0.20                     # 80% train, 20% test

# 5 different ways to split the data (different random shuffles)
SPLIT_SEEDS = [11, 22, 33, 44, 55]

# 4 different random seeds for XGBoost (so it behaves slightly differently each time)
MODEL_SEEDS = [101, 202, 303, 404]

# Settings for the XGBoost model
XGB_PARAMS = {
    "n_estimators": 500,         # Number of trees
    "max_depth": 6,              # How deep each tree can grow
    "learning_rate": 0.05,       # How much each new tree contributes
    "subsample": 0.8,            # Use 80% of rows for each tree
    "colsample_bytree": 0.8,     # Use 80% of features for each tree
    "objective": "reg:squarederror",
    "random_state": None,        # Will be filled in during the loop
    "n_jobs": -1                 # Use all CPU cores
}

# Save the full results table to CSV? (useful for later analysis)
SAVE_RESULTS_CSV = True
RESULTS_CSV_PATH = r"C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/Large Output/linear_xgb_multi_run_results.csv"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def compute_ci95(values):
    """
    Calculate mean, standard deviation, and 95% confidence interval
    for a list of numbers (used for R² scores across the 20 runs).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if n > 1 else 0.0
    ci95 = 1.96 * std_val / np.sqrt(n) if n > 1 else 0.0
    return mean_val, std_val, ci95


def plot_diag(ax, y_true, y_pred):
    """
    Draw a red dashed "perfect prediction" line (y = x) on a scatter plot.
    The line spans the full range of actual + predicted values.
    """
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)


def choose_representative_run(df_results, metric_col):
    """
    From all the runs of one model, pick the single run whose test R²
    is closest to the overall mean. This gives us a "typical" run to plot.
    """
    target = df_results[metric_col].mean()
    idx = (df_results[metric_col] - target).abs().idxmin()
    return df_results.loc[idx]


# ==========================================
# LOAD AND PREPARE DATA (only time = 0)
# ==========================================
# We only care about species abundances at the very beginning (time = 0)
# because that's what the other models (RNN/MLP) will also receive as input.

df1 = pd.read_csv(ABUNDANCE_CSV)   # Contains all time points and species abundances
df2 = pd.read_csv(OUTPUT_CSV)      # Contains the final production output for each community

# Clean column names (make everything lowercase and remove extra spaces)
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# Detect the column that identifies each community (sometimes called "community", sometimes "comm_name")
comm_col_1 = "community" if "community" in df1.columns else "comm_name"
comm_col_2 = "community" if "community" in df2.columns else "comm_name"

# Keep only the rows where time == 0
time_zero_df = df1[df1["time"] == 0].copy()

# Merge the abundance data with the final output value
merged_df = time_zero_df.merge(
    df2[[comm_col_2, "output"]],
    left_on=comm_col_1,
    right_on=comm_col_2,
    how="left"
).dropna(subset=["output"])   # Remove any communities missing an output value

# Find all the species columns (sp0, sp1, sp2, ...)
sp_columns = sorted(
    [col for col in merged_df.columns if col.startswith("sp")],
    key=lambda x: int(x[2:])          # Sort numerically: sp0, sp1, sp2...
)

# X = species abundances at time 0
# y = final production output
X = merged_df[sp_columns].values
y = merged_df["output"].values

# Print a nice summary so we know what we're working with
print("==========================================")
print("DATA SUMMARY")
print("==========================================")
print(f"Number of samples:   {X.shape[0]}")
print(f"Number of features:  {X.shape[1]}")
print(f"Output shape:        {y.shape}")
print(f"Species columns:     {len(sp_columns)}")
print(f"Train/Test split:    {int((1-TEST_SIZE)*100)}:{int(TEST_SIZE*100)}")
print(f"Split seeds:         {SPLIT_SEEDS}")
print(f"Model seeds:         {MODEL_SEEDS}")
print(f"Total runs:          {len(SPLIT_SEEDS) * len(MODEL_SEEDS)}")
print("==========================================\n")

# ==========================================
# RUN EXPERIMENTS (20 runs total)
# ==========================================
all_results = []                     # Will store every run's metrics
stored_predictions = {}              # Save predictions so we can plot later
run_id = 0

for split_seed in SPLIT_SEEDS:
    # Split the data into train and test using this random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=split_seed
    )

    for model_seed in MODEL_SEEDS:
        run_id += 1

        # -----------------------------
        # 1. Linear Regression
        # -----------------------------
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)

        lin_train_pred = lin_model.predict(X_train)
        lin_test_pred  = lin_model.predict(X_test)

        lin_train_r2 = r2_score(y_train, lin_train_pred)
        lin_test_r2  = r2_score(y_test, lin_test_pred)

        all_results.append({
            "run_id": run_id,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "model": "Linear",
            "train_r2": lin_train_r2,
            "test_r2": lin_test_r2
        })

        # Store predictions for later plotting
        stored_predictions[("Linear", run_id)] = {
            "y_train": y_train.copy(),
            "y_test": y_test.copy(),
            "train_pred": lin_train_pred.copy(),
            "test_pred": lin_test_pred.copy(),
            "split_seed": split_seed,
            "model_seed": model_seed
        }

        # -----------------------------
        # 2. XGBoost
        # -----------------------------
        xgb_params = XGB_PARAMS.copy()
        xgb_params["random_state"] = model_seed   # Make XGBoost reproducible

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)

        xgb_train_pred = xgb_model.predict(X_train)
        xgb_test_pred  = xgb_model.predict(X_test)

        xgb_train_r2 = r2_score(y_train, xgb_train_pred)
        xgb_test_r2  = r2_score(y_test, xgb_test_pred)

        all_results.append({
            "run_id": run_id,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "model": "XGBoost",
            "train_r2": xgb_train_r2,
            "test_r2": xgb_test_r2
        })

        stored_predictions[("XGBoost", run_id)] = {
            "y_train": y_train.copy(),
            "y_test": y_test.copy(),
            "train_pred": xgb_train_pred.copy(),
            "test_pred": xgb_test_pred.copy(),
            "split_seed": split_seed,
            "model_seed": model_seed
        }

# Convert all results into a nice pandas DataFrame
results_df = pd.DataFrame(all_results)

# ==========================================
# SAVE RESULTS (optional but recommended)
# ==========================================
if SAVE_RESULTS_CSV:
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Saved run results to:\n{RESULTS_CSV_PATH}\n")

# ==========================================
# SUMMARY METRICS (mean ± 95% CI)
# ==========================================
summary_rows = []

for model_name in ["Linear", "XGBoost"]:
    model_df = results_df[results_df["model"] == model_name]

    train_mean, train_std, train_ci = compute_ci95(model_df["train_r2"].values)
    test_mean, test_std, test_ci = compute_ci95(model_df["test_r2"].values)

    summary_rows.append({
        "model": model_name,
        "train_r2_mean": train_mean,
        "train_r2_std": train_std,
        "train_r2_ci95": train_ci,
        "test_r2_mean": test_mean,
        "test_r2_std": test_std,
        "test_r2_ci95": test_ci
    })

summary_df = pd.DataFrame(summary_rows)

print("==========================================")
print("SUMMARY ACROSS ALL RUNS")
print("==========================================")
for _, row in summary_df.iterrows():
    print(f"{row['model']}")
    print(f"  Train R2: {row['train_r2_mean']:.6f} ± {row['train_r2_ci95']:.6f} (95% CI)")
    print(f"  Test  R2: {row['test_r2_mean']:.6f} ± {row['test_r2_ci95']:.6f} (95% CI)")
    print(f"  Train STD: {row['train_r2_std']:.6f}")
    print(f"  Test  STD: {row['test_r2_std']:.6f}")
    print()

# ==========================================
# INTERPRETATION (simple rule of thumb)
# ==========================================
lin_test_mean = summary_df.loc[summary_df["model"] == "Linear", "test_r2_mean"].values[0]
xgb_test_mean = summary_df.loc[summary_df["model"] == "XGBoost", "test_r2_mean"].values[0]

print("INTERPRETATION:")
if xgb_test_mean > lin_test_mean + 0.05:
    print("Strong evidence of nonlinear interactions.")
elif abs(xgb_test_mean - lin_test_mean) < 0.02:
    print("Relationship appears mostly linear or weak.")
else:
    print("Some nonlinear structure may exist.")
print()

# ==========================================
# REPRESENTATIVE RUNS FOR SCATTER PLOTS
# Pick the run whose test R² is closest to the overall mean
# ==========================================
lin_runs = results_df[results_df["model"] == "Linear"].reset_index(drop=True)
xgb_runs = results_df[results_df["model"] == "XGBoost"].reset_index(drop=True)

lin_rep = choose_representative_run(lin_runs, "test_r2")
xgb_rep = choose_representative_run(xgb_runs, "test_r2")

lin_rep_run_id = int(lin_rep["run_id"])
xgb_rep_run_id = int(xgb_rep["run_id"])

lin_pred_pack = stored_predictions[("Linear", lin_rep_run_id)]
xgb_pred_pack = stored_predictions[("XGBoost", xgb_rep_run_id)]

# ==========================================
# PLOT 1: BAR PLOT OF MEAN R² WITH 95% CI
# ==========================================
models = ["Linear", "XGBoost"]

train_means = []
train_cis = []
test_means = []
test_cis = []

for model_name in models:
    row = summary_df[summary_df["model"] == model_name].iloc[0]
    train_means.append(row["train_r2_mean"])
    train_cis.append(row["train_r2_ci95"])
    test_means.append(row["test_r2_mean"])
    test_cis.append(row["test_r2_ci95"])

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, train_means, width, yerr=train_cis, capsize=6, label="Train")
plt.bar(x + width / 2, test_means,  width, yerr=test_cis,  capsize=6, label="Test")
plt.xticks(x, models)
plt.ylabel("R²")
plt.title("Mean Train/Test R² Across 20 Runs (95% CI)")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# PLOT 2: 2x2 SCATTER PLOTS (representative runs)
# ==========================================
plt.figure(figsize=(12, 10))

# TOP LEFT — Linear Regression (Train)
ax = plt.subplot(2, 2, 1)
ax.scatter(lin_pred_pack["y_train"], lin_pred_pack["train_pred"], alpha=0.6)
plot_diag(ax, lin_pred_pack["y_train"], lin_pred_pack["train_pred"])
ax.set_title(
    f"Linear (Train)\n"
    f"R² = {r2_score(lin_pred_pack['y_train'], lin_pred_pack['train_pred']):.4f}"
)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

# TOP RIGHT — Linear Regression (Test)
ax = plt.subplot(2, 2, 2)
ax.scatter(lin_pred_pack["y_test"], lin_pred_pack["test_pred"], alpha=0.6)
plot_diag(ax, lin_pred_pack["y_test"], lin_pred_pack["test_pred"])
ax.set_title(
    f"Linear (Test)\n"
    f"R² = {r2_score(lin_pred_pack['y_test'], lin_pred_pack['test_pred']):.4f}"
)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

# BOTTOM LEFT — XGBoost (Train)
ax = plt.subplot(2, 2, 3)
ax.scatter(xgb_pred_pack["y_train"], xgb_pred_pack["train_pred"], alpha=0.6)
plot_diag(ax, xgb_pred_pack["y_train"], xgb_pred_pack["train_pred"])
ax.set_title(
    f"XGBoost (Train)\n"
    f"R² = {r2_score(xgb_pred_pack['y_train'], xgb_pred_pack['train_pred']):.4f}"
)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

# BOTTOM RIGHT — XGBoost (Test)
ax = plt.subplot(2, 2, 4)
ax.scatter(xgb_pred_pack["y_test"], xgb_pred_pack["test_pred"], alpha=0.6)
plot_diag(ax, xgb_pred_pack["y_test"], xgb_pred_pack["test_pred"])
ax.set_title(
    f"XGBoost (Test)\n"
    f"R² = {r2_score(xgb_pred_pack['y_test'], xgb_pred_pack['test_pred']):.4f}"
)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

plt.tight_layout()
plt.show()

# ==========================================
# PLOT 3: BOX PLOT OF R² DISTRIBUTIONS
# ==========================================
plt.figure(figsize=(10, 6))

box_data = [
    results_df[results_df["model"] == "Linear"]["train_r2"].values,
    results_df[results_df["model"] == "Linear"]["test_r2"].values,
    results_df[results_df["model"] == "XGBoost"]["train_r2"].values,
    results_df[results_df["model"] == "XGBoost"]["test_r2"].values
]

labels = [
    "Linear\nTrain",
    "Linear\nTest",
    "XGBoost\nTrain",
    "XGBoost\nTest"
]

plt.boxplot(box_data, labels=labels)
plt.ylabel("R²")
plt.title("Distribution of R² Across 20 Runs")
plt.tight_layout()
plt.show()

# ==========================================
# OPTIONAL: PRINT RAW TABLES TO CONSOLE
# ==========================================
print("==========================================")
print("PER-RUN RESULTS (all 20 runs)")
print("==========================================")
print(results_df.to_string(index=False))

print("\n==========================================")
print("SUMMARY TABLE")
print("==========================================")
print(summary_df.to_string(index=False))

# ==========================================
# END OF SCRIPT
# ==========================================
# You now have:
#   • A clean CSV with every single run
#   • Summary statistics with confidence intervals
#   • Three publication-ready plots
#   • A simple interpretation of whether nonlinearity is present