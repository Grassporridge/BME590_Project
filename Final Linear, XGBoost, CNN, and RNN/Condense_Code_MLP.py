# ==========================================
# FULL PIPELINE (MLP VERSION) - 3-STEP APPROACH
# ==========================================
# Goal:
#   Predict microbial community dynamics over time and then predict
#   a final "production output" using those dynamics.
#
# Three-step pipeline:
#   1. Train a TimeSeriesMLP: Predict abundances at t=1..t20 from t=0 only
#   2. Train a FinalMLP: Learn to predict output from the TRUE full time-series (t0 + t1..t20)
#   3. Evaluate: Use the predicted time-series from Step 1 + FinalMLP from Step 2 → get R²
#
# This version uses plain Multi-Layer Perceptrons (MLPs) instead of RNNs for the time-series step.
# ==========================================

import time
import random
import ctypes
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# CONFIGURATION (All adjustable settings in one place)
# ==========================================
CONFIG = {
    "ABUNDANCE_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/simulated_abundance_switch.csv",
    "OUTPUT_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/production_output_summary.csv",
    "RESULTS_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/Large Output/MLP/three_step_pipeline.csv",

    "BATCH_SIZE": 64,          # Number of samples per training batch
    "EPOCHS": 2048,            # Maximum number of training epochs
    "LR": 1e-4,                # Learning rate for Adam optimizer

    "PATIENCE": 256,           # Early stopping: stop if no improvement for this many epochs
    "MIN_DELTA": 1e-5,         # Minimum improvement needed to reset patience counter

    "N_TIMESTEPS": 20,         # Predict t1 through t20 (20 future time points)

    "SEEDS": [0, 7, 15, 32, 126],   # Random seeds for reproducibility (5 runs per model size)

    # Hidden dimensions chosen so each model has roughly the target number of parameters
    "MODEL_CONFIGS": {
        "125K": 128,
        "250K": 192,
        "500K": 272,
        "750K": 320,
        "1000K": 384,
    }
}

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# SEED SETTING (for reproducible results)
# ==========================================
def set_seed(seed):
    """Set the same random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class TimeSeriesMLP(nn.Module):
    """
    Simple feed-forward MLP that takes species abundances at time t=0
    and predicts the abundances at t=1 through t=20 (all flattened into one vector).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class FinalMLP(nn.Module):
    """
    Final model that takes the FULL time-series (t=0 + t=1..t=20 flattened)
    and predicts the single production output value.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)          # Output is a single number
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# TRAINING FUNCTION (with Early Stopping)
# ==========================================
def train_model(model, loader, val_data, desc):
    """
    Train a PyTorch model using Adam optimizer and MSE loss.
    Includes early stopping based on validation loss.
    """
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    loss_fn = nn.MSELoss()

    X_val, y_val = val_data
    best_loss = float("inf")
    best_state = None
    patience = 0

    epoch_bar = tqdm(range(CONFIG["EPOCHS"]), desc=desc, leave=False)

    for epoch in epoch_bar:
        # === Training Phase ===
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(loader.dataset)

        # === Validation Phase ===
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()

        # Early stopping logic
        if val_loss < best_loss - CONFIG["MIN_DELTA"]:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        # Update progress bar
        epoch_bar.set_postfix({
            "train": f"{train_loss:.3e}",
            "val": f"{val_loss:.3e}",
            "best": f"{best_loss:.3e}",
            "pat": patience
        })

        if patience >= CONFIG["PATIENCE"]:
            break

    # Load the best model weights
    model.load_state_dict(best_state)
    return model

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    """
    Load abundance data and output data.
    Reshape into:
      - X_t0: abundances at time = 0
      - y_ts: abundances at t=1 to t=20 (flattened)
      - y_final: production output
    """
    df1 = pd.read_csv(CONFIG["ABUNDANCE_CSV"])
    df2 = pd.read_csv(CONFIG["OUTPUT_CSV"])

    # Clean column names
    df1.columns = df1.columns.str.lower().str.strip()
    df2.columns = df2.columns.str.lower().str.strip()

    y_final = df2["output"].values

    # Get all species columns (sp0, sp1, sp2, ...)
    sp_cols = [c for c in df1.columns if c.startswith("sp")]
    X_full = df1[sp_cols].values

    # Reshape: each sample has (N_TIMESTEPS + 1) time points
    n_steps = CONFIG["N_TIMESTEPS"] + 1
    n_samples = X_full.shape[0] // n_steps

    X_full = X_full[:n_samples * n_steps].reshape(n_samples, n_steps, -1)

    X_t0 = X_full[:, 0, :]                    # Only time = 0
    y_ts = X_full[:, 1:, :].reshape(n_samples, -1)  # t1 to t20 flattened

    return X_t0, y_ts, y_final[:n_samples]


def build_full(X_t0, y_ts_flat):
    """
    Reconstruct the full time-series as one long vector:
    [t0, t1, t2, ..., t20] for each sample.
    """
    n_samples, n_feat = X_t0.shape
    y_ts = y_ts_flat.reshape(n_samples, CONFIG["N_TIMESTEPS"], n_feat)
    full = np.concatenate([X_t0[:, None, :], y_ts], axis=1)
    return full.reshape(n_samples, -1)

# ==========================================
# MAIN PIPELINE (One complete run for a given hidden_dim and seed)
# ==========================================
def run_pipeline(X_t0, y_ts, y_final, hidden_dim, seed):
    """
    Run the full 3-step pipeline and return the test R² score.
    """
    # Split into train/test
    X_tr, X_te, yts_tr, yts_te, yf_tr, yf_te = train_test_split(
        X_t0, y_ts, y_final, test_size=0.2, random_state=seed
    )

    # Scale the data
    scaler_X = StandardScaler()
    scaler_ts = StandardScaler()

    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)

    yts_tr = scaler_ts.fit_transform(yts_tr)
    yts_te = scaler_ts.transform(yts_te)

    # Helper to move data to GPU/CPU
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).to(DEVICE)

    # ---------- STEP 1: Train TimeSeriesMLP ----------
    ts_model = TimeSeriesMLP(
        input_dim=X_tr.shape[1],
        hidden_dim=hidden_dim,
        output_dim=yts_tr.shape[1]
    ).to(DEVICE)

    loader = DataLoader(
        TensorDataset(to_tensor(X_tr), to_tensor(yts_tr)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True
    )

    ts_model = train_model(ts_model, loader, (to_tensor(X_te), to_tensor(yts_te)), "TS Model")

    # ---------- STEP 2: Train FinalMLP on TRUE full time-series ----------
    X_full_tr = build_full(X_tr, scaler_ts.inverse_transform(yts_tr))
    X_full_tr = StandardScaler().fit_transform(X_full_tr)

    final_model = FinalMLP(X_full_tr.shape[1], hidden_dim).to(DEVICE)

    loader2 = DataLoader(
        TensorDataset(to_tensor(X_full_tr), to_tensor(yf_tr.reshape(-1, 1))),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True
    )

    final_model = train_model(final_model, loader2,
                              (to_tensor(X_full_tr), to_tensor(yf_tr.reshape(-1, 1))), "Final Model")

    # ---------- STEP 3: Evaluate using predicted time-series ----------
    with torch.no_grad():
        ts_pred = ts_model(to_tensor(X_te)).cpu().numpy()

    ts_pred = scaler_ts.inverse_transform(ts_pred)

    X_full_te = build_full(X_te, ts_pred)
    X_full_te = StandardScaler().fit_transform(X_full_te)

    with torch.no_grad():
        pred = final_model(to_tensor(X_full_te)).cpu().numpy().flatten()

    return r2_score(yf_te, pred)

# ==========================================
# ANTI-SLEEP UTILITIES (prevents computer from sleeping during long training)
# ==========================================
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """Tell Windows to keep the computer awake."""
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_sleep():
    """Restore normal sleep behavior."""
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    prevent_sleep()

    try:
        # Load all data once
        X_t0, y_ts, y_final = load_data()

        results = []

        # Test different model sizes
        for name, hidden_dim in CONFIG["MODEL_CONFIGS"].items():
            r2_list = []
            time_list = []

            print(f"\n=== Training models with size: {name} ===")

            for seed in CONFIG["SEEDS"]:
                set_seed(seed)

                start = time.time()
                r2 = run_pipeline(X_t0, y_ts, y_final, hidden_dim, seed)
                runtime = time.time() - start

                r2_list.append(r2)
                time_list.append(runtime)

                print(f"{name} | Seed {seed} | R2: {r2:.4f} | Time: {runtime:.2f}s")

            # Compute mean and 95% confidence interval
            r2_mean = np.mean(r2_list)
            r2_std = np.std(r2_list)
            r2_ci = 1.96 * r2_std / np.sqrt(len(r2_list))

            time_mean = np.mean(time_list)
            time_std = np.std(time_list)
            time_ci = 1.96 * time_std / np.sqrt(len(time_list))

            print("==============================")
            print(f"MODEL: {name}")
            print(f"R2 Mean: {r2_mean:.4f} ± {r2_ci:.4f} (95% CI)")
            print(f"Time Mean: {time_mean:.2f}s ± {time_ci:.2f}s (95% CI)")
            print("==============================")

            results.append({
                "model": name,
                "r2_mean": r2_mean,
                "r2_ci": r2_ci,
                "time_mean": time_mean,
                "time_ci": time_ci
            })

        # Save final results
        pd.DataFrame(results).to_csv(CONFIG["RESULTS_CSV"], index=False)
        print(f"\nResults saved to: {CONFIG['RESULTS_CSV']}")

    finally:
        allow_sleep()


if __name__ == "__main__":
    main()