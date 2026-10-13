"""Training, early stopping, inverse transforms, and volatility metrics."""
import copy
import math
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from models import MultiLayerLiGRU, MultiLayerGRU

def train_and_evaluate(data, hidden_dim=8, num_layers=2, dropout=0.2,
                       lr=5e-4, batch_size=64, num_epochs=300, seed=0,
                       model_type="ligru", patience=None, min_delta=1e-4,
                       restore_best_weights=True, verbose=False, print_every=10):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    model_cls = MultiLayerLiGRU if model_type.lower() == "ligru" else MultiLayerGRU
    model = model_cls(X_train.shape[-1], hidden_dim, num_layers, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state, best_epoch, no_improve = None, 0, 0
    started = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred[:, -1, :], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val)
            val_loss = criterion(val_pred[:, -1, :], y_val).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - min_delta:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            if restore_best_weights:
                best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1

        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d} | train RMSE={math.sqrt(train_loss):.4f} "
                  f"| val RMSE={math.sqrt(val_loss):.4f}")
        if patience is not None and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}; best epoch={best_epoch+1}")
            break

    elapsed = time.perf_counter() - started
    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        def predict(X):
            return model(X)[0][:, -1, :].cpu().numpy()
        pred_train, pred_val, pred_test = predict(X_train), predict(X_val), predict(X_test)

    scaler = data["scaler_target"]
    pred_train_inv = scaler.inverse_transform(pred_train)
    pred_val_inv = scaler.inverse_transform(pred_val)
    pred_test_inv = scaler.inverse_transform(pred_test)
    y_train_inv = scaler.inverse_transform(y_train.numpy())
    y_val_inv = scaler.inverse_transform(y_val.numpy())
    y_test_inv = scaler.inverse_transform(y_test.numpy())

    # Convert log(r^2 + eps) predictions back to a positive squared-return proxy.
    idx = data["target_indices"]
    test_start = data["train_size"] + data["val_size"]
    test_indices = idx[test_start:]
    true_sq = data["returns"][test_indices] ** 2
    delta = 1e-6
    pred_sq = np.maximum(np.exp(pred_test_inv.ravel()) - data["eps"], delta)
    true_sq = np.maximum(true_sq, delta)
    qlike = float(np.mean(true_sq / pred_sq - np.log(true_sq / pred_sq) - 1.0))
    mse_sq = float(np.mean((true_sq - pred_sq) ** 2))
    test_loss = float(np.mean((pred_test.ravel() - y_test.numpy().ravel()) ** 2))

    return {
        "model": model, "train_losses": train_losses, "val_losses": val_losses,
        "train_pred_inv": pred_train_inv, "val_pred_inv": pred_val_inv,
        "test_pred_inv": pred_test_inv, "y_train_inv": y_train_inv,
        "y_val_inv": y_val_inv, "y_test_inv": y_test_inv,
        "train_rmse": float(np.sqrt(np.mean((pred_train.ravel()-y_train.numpy().ravel())**2))),
        "val_rmse": float(np.sqrt(np.mean((pred_val.ravel()-y_val.numpy().ravel())**2))),
        "test_rmse": float(np.sqrt(test_loss)), "test_loss": test_loss,
        "mse_sq_scale": mse_sq, "qlike": qlike,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_time": elapsed, "best_epoch": best_epoch,
        "epochs_run": len(train_losses), "stopped_early": len(train_losses) < num_epochs,
    }

def run_multiple_seeds(data, n_seeds=12, **train_kwargs):
    """Run a fixed configuration across seeds and return summary and full results."""
    import pandas as pd
    rows, results = [], {}
    for seed in range(n_seeds):
        result = train_and_evaluate(data, seed=seed, **train_kwargs)
        results[seed] = result
        rows.append({
            "seed": seed, "train_rmse": result["train_rmse"],
            "val_rmse": result["val_rmse"], "test_rmse": result["test_rmse"],
            "test_loss": result["test_loss"], "test_qlike": result["qlike"],
            "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"] + 1,
        })
        print(f"seed={seed:02d}: train RMSE={result['train_rmse']:.4f}, "
              f"val RMSE={result['val_rmse']:.4f}, test RMSE={result['test_rmse']:.4f}, "
              f"QLIKE={result['qlike']:.4f}")
    return pd.DataFrame(rows), results
