# Hybrid EGARCH–Modified LiGRU volatility forecasting

This package consolidates repeated notebook code into reusable modules.

## Structure

- `data.py`: reads the crisis CSVs, constructs return and EGARCH features, creates lookback sequences, prevents windows from crossing crisis boundaries, and makes chronological splits.
- `models.py`: defines the modified LiGRU cell/network and a standard GRU baseline.
- `training.py`: contains shared training, early stopping, evaluation, positive squared-return proxy conversion, QLIKE, and multi-seed runs.
- `plotting.py`: reusable return diagnostics, ACF, loss, and prediction-fit plots.
- `diagnostics.py`: descriptive statistics, ADF/KPSS, Ljung–Box, and ARCH-LM diagnostics.
- `experiments.py`: named configurations for hybrid, returns-only, historical-only, and GRU runs.
- `main.py`: command-line entry point.
- `relabel_plot.py`: separate image-label utility.

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Put these CSVs in `data/` (or beside `main.py`):
- `S&P 500 Historical Data 1987.csv`
- `S&P 500 Historical Data 2000-2002.csv`
- `S&P 500 Historical Data 2008.csv`
- `S&P 500 Historical Data 2020-2022.csv`

Each CSV must have a `Change %` column, like the original notebook.

## Run examples

```bash
# Hybrid EGARCH-feature LiGRU with all four crisis periods
python main.py --experiment hybrid_4_periods --plots

# Returns-only LiGRU
python main.py --experiment returns_only_4_periods --plots

# Historical-only hybrid, excluding 2020–2022
python main.py --experiment hybrid_historical --plots

# Standard GRU comparison
python main.py --experiment gru_baseline --plots

# Run 15 random seeds and save a seed-results CSV
python main.py --experiment hybrid_4_periods --seeds 15

# Return distribution, ACF, and descriptive statistics
python main.py --experiment hybrid_4_periods --diagnostics
```

Outputs are written to `outputs/`.

## Important fidelity notes

This is a structured, reusable implementation of the core experiments represented in the pasted 8,483-line text. It consolidates repeated LiGRU/hybrid training and plotting code rather than preserving every copied cell verbatim. The source also contains several separately copied ablation runs, commented-out grid searches, a custom image relabeling routine, and quantum/classical gate-map benchmarking code. Those are not all automatically executed by this entry point. The quantum benchmark should remain a separate optional module because it has extra dependencies (PennyLane and/or TorchQuantum) and a different purpose from the volatility forecasting pipeline.

Before using outputs in a paper, compare this modular implementation's split/scaling conventions and reported metrics with the exact notebook version you intend to report. The source uses both full-period and historical-only datasets in different sections, and those should be treated as separate experiments.
