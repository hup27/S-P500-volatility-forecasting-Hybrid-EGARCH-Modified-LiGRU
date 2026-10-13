"""Run a named LiGRU/GRU volatility forecasting experiment."""
import argparse
from pathlib import Path
import pandas as pd
from data import load_returns, build_dataset
from training import train_and_evaluate, run_multiple_seeds
from plotting import plot_loss_history, plot_predictions, plot_return_diagnostics, plot_acf_diagnostics
from diagnostics import descriptive_statistics
from experiments import CONFIGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(CONFIGS), default="hybrid_4_periods")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds; 1 runs a single seed.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.experiment].copy()
    returns, crisis_id, crisis_names = load_returns(args.data_dir, include_covid=cfg.pop("include_covid"))
    data = build_dataset(returns, crisis_id, seq_length=cfg.pop("seq_length"),
                         eps=cfg.pop("eps"), use_egarch=cfg.pop("use_egarch"))
    if args.diagnostics:
        print("\nDescriptive statistics:")
        print(descriptive_statistics(returns, crisis_id, crisis_names).round(4))
        plot_return_diagnostics(returns)
        plot_acf_diagnostics(returns)

    if args.seeds > 1:
        summary, results = run_multiple_seeds(data, n_seeds=args.seeds, **cfg)
        summary.to_csv(f"{args.experiment}_seed_results.csv", index=False)
        best_seed = int(summary["val_rmse"].idxmin())
        result = results[best_seed]
        print("\nSaved seed metrics.")
    else:
        result = train_and_evaluate(data, seed=42, **cfg)

    print(f"\nExperiment: {args.experiment}")
    for metric in ("train_rmse", "val_rmse", "test_rmse", "test_loss", "mse_sq_scale", "qlike",
                   "n_params", "train_time", "epochs_run", "best_epoch"):
        print(f"{metric:16s}: {result[metric]}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.DataFrame({
        "true_vol_proxy": __import__("numpy").concatenate(
            [result["y_train_inv"].ravel(), result["y_val_inv"].ravel(), result["y_test_inv"].ravel()]),
        "predicted_vol": __import__("numpy").concatenate(
            [result["train_pred_inv"].ravel(), result["val_pred_inv"].ravel(), result["test_pred_inv"].ravel()]),
        "split": (["train"] * len(result["y_train_inv"]) +
                  ["val"] * len(result["y_val_inv"]) + ["test"] * len(result["y_test_inv"]))
    }).to_csv(out / f"{args.experiment}_predictions.csv", index=False)
    pd.DataFrame({"epoch": range(1, len(result["train_losses"])+1),
                  "train_loss": result["train_losses"],
                  "val_loss": result["val_losses"]}).to_csv(out / f"{args.experiment}_loss_history.csv", index=False)
    if args.plots:
        plot_loss_history(result, title=f"{args.experiment}: loss")
        plot_predictions(result, title=f"{args.experiment}: true vs predicted", 
                         save_path=out / f"{args.experiment}_prediction_fit.png")
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()
