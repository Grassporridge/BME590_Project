# ==========================================
# FULL PIPELINE (RNN + PARAMETER CONTROL)
# ==========================================
# Goal: Predict a microbial time-series (t0 → t1..t20) using an RNN,
# then use the full predicted time-series to predict a final "production output".
#
# Three-step process:
#   1. Train RNN to predict future abundances from t=0
#   2. Train a final MLP on the TRUE full time-series → output
#   3. Evaluate: use RNN predictions + final MLP → get R² on test data
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
# CONFIGURATION (all the settings you can change in one place)
# ==========================================
CONFIG = {
    "ABUNDANCE_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/simulated_abundance_switch.csv",
    "OUTPUT_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/production_output_summary.csv",
    "RESULTS_CSV": "C:/Users/josep/Downloads/Python_3.14.2/AI and Biocomputing/Project/Large Output/RNN/three_step_pipeline_rnn.csv",

    "BATCH_SIZE": 64,          # How many samples per training batch
    "EPOCHS": 2048,            # Maximum number of training epochs
    "LR": 1e-4,                # Learning rate for Adam optimizer

    "PATIENCE": 256,           # Early stopping patience
    "MIN_DELTA": 1e-5,         # Minimum improvement to reset patience

    "N_TIMESTEPS": 20,         # Number of future time points to predict (t1 to t20)

    "SEEDS": [0, 7, 15, 32, 126],   # Different random seeds for reproducibility

    # Different model sizes (parameter budgets) we will test
    "MODEL_CONFIGS": {
        "125K": 125_000,
        "250K": 250_000,
        "500K": 500_000,
        "750K": 750_000,
        "1000K": 1_000_000,
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# SEED SETTING (makes results reproducible)
# ==========================================
def set_seed(seed):
    """Set the same random seed everywhere so experiments can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# PARAMETER COUNT UTILITIES
# ==========================================
def count_params(model):
    """Count how many trainable parameters a PyTorch model has."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_hidden_dim(input_dim, output_dim, target_params):
    """
    Binary search to find the largest hidden dimension that keeps
    the total number of parameters under the target budget.
    """
    low, high = 8, 1024
    best = low
    for _ in range(20):
        mid = (low + high) // 2
        test_model = TimeSeriesRNN(input_dim, mid, output_dim)
        params = count_params(test_model)
        if params < target_params:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class TimeSeriesRNN(nn.Module):
    """
    RNN (GRU) that takes abundances at t=0 and predicts abundances
    at t=1 through t=20 (flattened).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # Add sequence dimension: (batch, 1, features)
        out, _ = self.rnn(x)
        out = out[:, -1, :]         # Take the last time step
        return self.fc(out)


class FinalMLP(nn.Module):
    """
    Simple feed-forward network that takes the FULL time-series
    (t0 + t1..t20 flattened) and predicts the final production output.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# TRAINING LOOP (with early stopping)
# ==========================================
def train_model(model, loader, val_data, desc):
    """Train a model with Adam + MSE loss and early stopping on validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    loss_fn = nn.MSELoss()

    X_val, y_val = val_data
    best_loss = float("inf")
    best_state = None
    patience = 0

    bar = tqdm(range(CONFIG["EPOCHS"]), desc=desc, leave=False)

    for epoch in bar:
        model.train()
        total = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        train_loss = total / len(loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        # Early stopping logic
        if val_loss < best_loss - CONFIG["MIN_DELTA"]:
            best_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        bar.set_postfix({
            "train": f"{train_loss:.2e}",
            "val": f"{val_loss:.2e}",
            "pat": patience
        })

        if patience >= CONFIG["PATIENCE"]:
            break

    model.load_state_dict(best_state)
    return model

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_data():
    """Load abundance CSV and output CSV, reshape into (samples, timesteps, features)."""
    df1 = pd.read_csv(CONFIG["ABUNDANCE_CSV"])
    df2 = pd.read_csv(CONFIG["OUTPUT_CSV"])

    df1.columns = df1.columns.str.lower().str.strip()
    df2.columns = df2.columns.str.lower().str.strip()

    y_final = df2["output"].values

    sp_cols = [c for c in df1.columns if c.startswith("sp")]
    X_full = df1[sp_cols].values

    n_steps = CONFIG["N_TIMESTEPS"] + 1          # t0 + t1..t20
    n_samples = X_full.shape[0] // n_steps

    # Reshape to (samples, timesteps, species)
    X_full = X_full[:n_samples*n_steps].reshape(n_samples, n_steps, -1)

    X_t0 = X_full[:, 0, :]                        # Only time = 0
    y_ts = X_full[:, 1:, :].reshape(n_samples, -1)  # t1..t20 flattened

    return X_t0, y_ts, y_final[:n_samples]


def build_full(X_t0, y_ts_flat):
    """Reconstruct the full time-series (t0 + predicted t1..t20) as one long vector."""
    n_samples, n_feat = X_t0.shape
    y_ts = y_ts_flat.reshape(n_samples, CONFIG["N_TIMESTEPS"], n_feat)
    full = np.concatenate([X_t0[:, None, :], y_ts], axis=1)
    return full.reshape(n_samples, -1)

# ==========================================
# MAIN PIPELINE (one run for a given parameter budget + seed)
# ==========================================
def run_pipeline(X_t0, y_ts, y_final, param_budget, seed):
    """Run the full 3-step pipeline and return test R²."""
    # Split data
    X_tr, X_te, yts_tr, yts_te, yf_tr, yf_te = train_test_split(
        X_t0, y_ts, y_final, test_size=0.2, random_state=seed
    )

    # Scale inputs
    scaler_X = StandardScaler()
    scaler_ts = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)
    yts_tr = scaler_ts.fit_transform(yts_tr)
    yts_te = scaler_ts.transform(yts_te)

    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).to(DEVICE)

    # ---------- STEP 1: Train RNN to predict future abundances ----------
    input_dim = X_tr.shape[1]
    output_dim = yts_tr.shape[1]
    hidden_dim = find_hidden_dim(input_dim, output_dim, param_budget)

    ts_model = TimeSeriesRNN(input_dim, hidden_dim, output_dim).to(DEVICE)
    print(f"TS Params: {count_params(ts_model):,}")

    loader = DataLoader(
        TensorDataset(to_tensor(X_tr), to_tensor(yts_tr)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True
    )
    ts_model = train_model(ts_model, loader, (to_tensor(X_te), to_tensor(yts_te)), "TS RNN")

    # ---------- STEP 2: Train Final MLP on TRUE full time-series ----------
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

    # ---------- STEP 3: Evaluate on test set using RNN predictions ----------
    with torch.no_grad():
        ts_pred = ts_model(to_tensor(X_te)).cpu().numpy()
    ts_pred = scaler_ts.inverse_transform(ts_pred)

    X_full_te = build_full(X_te, ts_pred)
    X_full_te = StandardScaler().fit_transform(X_full_te)

    with torch.no_grad():
        pred = final_model(to_tensor(X_full_te)).cpu().numpy()

    return r2_score(yf_te, pred)

# ==========================================
# ANTI-SLEEP (prevents Windows from sleeping during long runs)
# ==========================================
def prevent_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

def allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    prevent_sleep()
    try:
        X_t0, y_ts, y_final = load_data()
        results = []

        for name, budget in CONFIG["MODEL_CONFIGS"].items():
            r2_list = []
            time_list = []

            for seed in CONFIG["SEEDS"]:
                set_seed(seed)
                start = time.time()
                r2 = run_pipeline(X_t0, y_ts, y_final, budget, seed)
                runtime = time.time() - start

                r2_list.append(r2)
                time_list.append(runtime)
                print(f"{name} | Seed {seed} | R2: {r2:.4f} | Time: {runtime:.2f}s")

            # Compute mean + 95% CI
            r2_mean = np.mean(r2_list)
            r2_ci = 1.96 * np.std(r2_list) / np.sqrt(len(r2_list))
            time_mean = np.mean(time_list)
            time_ci = 1.96 * np.std(time_list) / np.sqrt(len(time_list))

            print(f"{name} | R2: {r2_mean:.4f} ± {r2_ci:.4f}")

            results.append({
                "model": name,
                "r2_mean": r2_mean,
                "r2_ci": r2_ci,
                "time_mean": time_mean,
                "time_ci": time_ci
            })

        pd.DataFrame(results).to_csv(CONFIG["RESULTS_CSV"], index=False)

    finally:
        allow_sleep()


if __name__ == "__main__":
    main()