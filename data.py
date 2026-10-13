"""Data loading, feature construction, and sequence preparation for volatility experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from arch import arch_model

DEFAULT_FILES = {
    "1987": "S&P 500 Historical Data 1987.csv",
    "2000-2002": "S&P 500 Historical Data 2000-2002.csv",
    "2008": "S&P 500 Historical Data 2008.csv",
    "2020-2022": "S&P 500 Historical Data 2020-2022.csv",
}

def load_returns(data_dir="data", include_covid=True):
    """Load crisis CSVs and return demeaned decimal returns plus aligned crisis labels."""
    names = list(DEFAULT_FILES) if include_covid else ["1987", "2000-2002", "2008"]
    pieces = []
    for crisis in names:
        path = Path(data_dir) / DEFAULT_FILES[crisis]
        if not path.exists():
            # Also allow CSVs to sit beside the Python files.
            path = Path(DEFAULT_FILES[crisis])
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {DEFAULT_FILES[crisis]!r}. Put it in {data_dir!r} or the project root."
            )
        df = pd.read_csv(path)
        if "Change %" not in df.columns:
            raise ValueError(f"{path.name} is missing required column 'Change %'.")
        pct = pd.to_numeric(df["Change %"].astype("string").str.rstrip("%"), errors="coerce")
        piece = pd.DataFrame({"return_pct": pct, "crisis": crisis}).dropna(subset=["return_pct"])
        pieces.append(piece)
    combined = pd.concat(pieces, ignore_index=True)
    returns = combined["return_pct"].to_numpy(dtype=np.float64) / 100.0
    returns -= returns.mean()
    return returns, combined["crisis"].to_numpy(), names

def find_crisis_boundaries(crisis_id):
    return [i for i in range(1, len(crisis_id)) if crisis_id[i] != crisis_id[i - 1]]

def create_sequences(ret_feature, second_feature, target, seq_length, crisis_id):
    """Create lookback windows, excluding any window/target crossing a crisis seam."""
    xs, ys, starts = [], [], []
    for i in range(len(ret_feature) - seq_length):
        labels = crisis_id[i:i + seq_length + 1]
        if len(set(labels)) != 1:
            continue
        cols = [np.asarray(ret_feature)[i:i + seq_length]]
        if second_feature is not None:
            cols.append(np.asarray(second_feature)[i:i + seq_length])
        xs.append(np.column_stack(cols))
        ys.append(target[i + seq_length])
        starts.append(i)
    n_features = 1 + (second_feature is not None)
    return (np.asarray(xs, dtype=np.float32).reshape(-1, seq_length, n_features),
            np.asarray(ys, dtype=np.float32).reshape(-1),
            np.asarray(starts, dtype=int))

def verify_no_boundary_crossing(start_idxs, crisis_id, seq_length):
    boundaries = find_crisis_boundaries(crisis_id)
    violations = [(int(i), int(b)) for i in start_idxs for b in boundaries
                  if i < b <= i + seq_length]
    if violations:
        raise AssertionError(f"{len(violations)} sequences cross crisis boundaries: {violations[:5]}")
    return boundaries, violations

def build_dataset(returns, crisis_id, seq_length=20, eps=1e-6, use_egarch=True,
                  train_fraction=0.70, val_fraction=0.15, verify_boundaries=True):
    """Build scaled features/target and chronological train/validation/test tensors.

    Scalers are fitted on the initial chronological 70% of raw observations only.
    Sequence splits are then chronological 70/15/15, matching the source workflow.
    """
    if len(returns) != len(crisis_id):
        raise ValueError("returns and crisis_id must have equal lengths.")
    raw_train_end = int(len(returns) * train_fraction)
    scaler_ret = RobustScaler().fit(returns[:raw_train_end].reshape(-1, 1))
    ret_scaled = scaler_ret.transform(returns.reshape(-1, 1)).ravel()

    egarch_scaled = None
    scaler_egarch = None
    if use_egarch:
        returns_pct = returns * 100.0
        am = arch_model(returns_pct, vol="EGARCH", p=1, o=1, q=1,
                        dist="normal", mean="Zero")
        fitted = am.fit(disp="off", last_obs=raw_train_end)
        fixed = am.fix(fitted.params)
        sigma = np.asarray(fixed.conditional_volatility, dtype=float) / 100.0
        egarch_log = np.log(sigma ** 2 + 1e-6)
        scaler_egarch = RobustScaler().fit(egarch_log[:raw_train_end].reshape(-1, 1))
        egarch_scaled = scaler_egarch.transform(egarch_log.reshape(-1, 1)).ravel()

    target_raw = np.log(returns ** 2 + eps)
    scaler_target = StandardScaler().fit(target_raw[:raw_train_end].reshape(-1, 1))
    target_scaled = scaler_target.transform(target_raw.reshape(-1, 1)).ravel()

    X, y, starts = create_sequences(ret_scaled, egarch_scaled, target_scaled, seq_length, crisis_id)
    if verify_boundaries:
        verify_no_boundary_crossing(starts, crisis_id, seq_length)

    n = len(X)
    train_n = int(n * train_fraction)
    val_n = int(n * val_fraction)
    a, b = train_n, train_n + val_n
    def xt(a_, b_):
        return torch.tensor(X[a_:b_], dtype=torch.float32)
    def yt(a_, b_):
        return torch.tensor(y[a_:b_], dtype=torch.float32).unsqueeze(-1)

    # Align squared-return observations with target indices, not with the sequence start.
    target_indices = starts + seq_length
    data = {
        "X_train": xt(0, a), "y_train": yt(0, a),
        "X_val": xt(a, b), "y_val": yt(a, b),
        "X_test": xt(b, n), "y_test": yt(b, n),
        "scaler_target": scaler_target, "scaler_ret": scaler_ret,
        "scaler_egarch": scaler_egarch, "returns": returns,
        "crisis_id": crisis_id, "target_raw": target_raw,
        "target_indices": target_indices, "start_idxs": starts,
        "train_size": a, "val_size": b-a, "seq_length": seq_length,
        "use_egarch": use_egarch, "eps": eps,
    }
    return data
