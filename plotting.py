"""Reusable plots for diagnostics, losses, and fitted volatility proxies."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

def plot_return_diagnostics(returns):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(returns, bins=60, density=True, alpha=0.7)
    xs = np.linspace(np.min(returns), np.max(returns), 200)
    axes[0].plot(xs, stats.norm.pdf(xs, np.mean(returns), np.std(returns)), "r--", label="Normal fit")
    axes[0].set_title("Return distribution"); axes[0].set_xlabel("Return"); axes[0].legend()
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot vs Normal")
    plt.tight_layout()
    return fig, axes

def plot_acf_diagnostics(returns, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(returns, lags=lags, ax=axes[0]); axes[0].set_title("ACF — Returns")
    plot_acf(returns**2, lags=lags, ax=axes[1]); axes[1].set_title("ACF — Squared Returns")
    plt.tight_layout()
    return fig, axes

def plot_loss_history(result, title="LiGRU training and validation loss"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["train_losses"], label="Train Loss")
    ax.plot(result["val_losses"], label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig, ax

def plot_predictions(result, title="True vs predicted log squared returns", save_path=None):
    actual = np.concatenate([result["y_train_inv"].ravel(), result["y_val_inv"].ravel(),
                             result["y_test_inv"].ravel()])
    predicted = np.concatenate([result["train_pred_inv"].ravel(), result["val_pred_inv"].ravel(),
                                result["test_pred_inv"].ravel()])
    train_end = len(result["y_train_inv"])
    val_end = train_end + len(result["y_val_inv"])
    fig, ax = plt.subplots(figsize=(18, 5), dpi=180)
    ax.plot(actual, label="True log squared return", linewidth=1)
    ax.plot(predicted, label="Predicted log squared return", linewidth=1.2, alpha=0.85)
    ax.axvspan(0, train_end, alpha=0.07, label="Train")
    ax.axvspan(train_end, val_end, alpha=0.07, label="Validation")
    ax.axvspan(val_end, len(actual), alpha=0.07, label="Test")
    ax.axvline(train_end, linestyle="--", linewidth=1)
    ax.axvline(val_end, linestyle="--", linewidth=1)
    ax.set_xlabel("Chronological sequence"); ax.set_ylabel("Log(r² + epsilon)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
