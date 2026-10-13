"""Named experiment configurations corresponding to the repeated notebook variants."""
CONFIGS = {
    # Original 1987 + 2000–2002 + 2008 + 2020–2022 hybrid EGARCH feature run.
    "hybrid_4_periods": {
        "include_covid": True, "use_egarch": True, "seq_length": 20, "eps": 1e-6,
        "hidden_dim": 8, "num_layers": 2, "dropout": 0.2,
        "lr": 5e-4, "batch_size": 64, "num_epochs": 300,
        "patience": None, "model_type": "ligru",
    },
    # Returns-only LiGRU variants in the source notebook.
    "returns_only_4_periods": {
        "include_covid": True, "use_egarch": False, "seq_length": 20, "eps": 1e-6,
        "hidden_dim": 8, "num_layers": 2, "dropout": 0.2,
        "lr": 5e-4, "batch_size": 64, "num_epochs": 300,
        "patience": None, "model_type": "ligru",
    },
    # Historical-only runs that explicitly exclude 2020–2022.
    "returns_only_historical": {
        "include_covid": False, "use_egarch": False, "seq_length": 20, "eps": 1e-6,
        "hidden_dim": 8, "num_layers": 2, "dropout": 0.2,
        "lr": 5e-4, "batch_size": 64, "num_epochs": 300,
        "patience": None, "model_type": "ligru",
    },
    "hybrid_historical": {
        "include_covid": False, "use_egarch": True, "seq_length": 20, "eps": 1e-6,
        "hidden_dim": 8, "num_layers": 2, "dropout": 0.2,
        "lr": 5e-4, "batch_size": 64, "num_epochs": 300,
        "patience": None, "model_type": "ligru",
    },
    "gru_baseline": {
        "include_covid": True, "use_egarch": True, "seq_length": 20, "eps": 1e-6,
        "hidden_dim": 6, "num_layers": 2, "dropout": 0.4,
        "lr": 5e-4, "batch_size": 64, "num_epochs": 300,
        "patience": None, "model_type": "gru",
    },
}
