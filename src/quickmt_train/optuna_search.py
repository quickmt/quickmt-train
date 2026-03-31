import sys
import os
from datetime import datetime

try:
    import optuna
except ImportError:
    print("Please install optuna to use this script: pip install optuna")
    sys.exit(1)

try:
    from quickmt_train.config import load_config
except ImportError:
    # Append the src directory to sys.path if run directly as a script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from quickmt_train.config import load_config


def objective(trial, args):
    model_cfg, data_cfg, train_cfg, export_cfg = load_config(args.config)

    # Override config with args
    train_cfg.max_steps = args.max_steps
    train_cfg.eval_steps = args.eval_steps
    train_cfg.quick_test_samples = 0
    train_cfg.save_checkpoints = False
    train_cfg.aim_repo = None

    # Do not enable compilation during hyperparameter search (can be slow/flaky across trials)
    train_cfg.enable_torch_compile = False

    # Suggest hyperparameters
    train_cfg.lr = trial.suggest_float("lr", *args.lr_range, log=True)
    train_cfg.weight_decay = trial.suggest_float(
        "weight_decay", *args.weight_decay_range, log=True
    )
    train_cfg.grad_clip = trial.suggest_float("grad_clip", *args.grad_clip_range)
    train_cfg.label_smoothing = trial.suggest_float(
        "label_smoothing", *args.label_smoothing_range
    )
    model_cfg.dropout = trial.suggest_float("dropout", *args.dropout_range)
    model_cfg.norm_type = trial.suggest_categorical("norm_type", args.norm_types)
    model_cfg.activation = trial.suggest_categorical("activation", args.activations)
    model_cfg.layernorm_eps = trial.suggest_float(
        "layernorm_eps", *args.layernorm_eps_range, log=True
    )
    model_cfg.ff_bias = trial.suggest_categorical("ff_bias", args.ff_biases)

    # Suggest corpus weights if requested
    if args.tune_corpus_weights:
        for i, corpus in enumerate(data_cfg.corpora):
            corpus.weight = trial.suggest_int(f"corpus_{i + 1}_weight", 0, 10)

    from quickmt_train.train import train

    def get_time_info():
        curr_time = datetime.now().strftime("%H:%M:%S")
        return f"[{curr_time}] [Trial {trial.number}]"

    def on_eval_step(val_metrics, step):
        metric_val = val_metrics.get(args.metric)
        if metric_val is None:
            raise ValueError(
                f"Metric '{args.metric}' not found in validation metrics {list(val_metrics.keys())}"
            )

        # Report intermediate evaluation to Optuna functionality
        trial.report(metric_val, step)

        # Optuna handles the pruning mechanism
        if trial.should_prune():
            print(
                f"{get_time_info()} Trial pruned at step {step} (Metric: {metric_val:.4f})"
            )
            raise optuna.TrialPruned()

    latest_metrics = train(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        on_eval_step=on_eval_step,
    )

    if latest_metrics is None:
        raise RuntimeError(
            "Training finished without returning any validation metrics."
        )

    metric_val = latest_metrics.get(args.metric)
    if metric_val is None:
        raise ValueError(
            f"Metric '{args.metric}' not found in validation metrics {list(latest_metrics.keys())}"
        )

    return metric_val


def search_cli(
    config: str,
    metric: str = "loss",
    max_steps: int = 1000,
    eval_steps: int = 200,
    n_trials: int = 200,
    study_name: str = None,
    db: str = None,
    lr_range: tuple[float, float] = (1e-5, 5e-3),
    weight_decay_range: tuple[float, float] = (1e-4, 0.1),
    grad_clip_range: tuple[float, float] = (0.1, 5.0),
    label_smoothing_range: tuple[float, float] = (0.0, 0.2),
    dropout_range: tuple[float, float] = (0.0, 0.3),
    norm_types: tuple[str, ...] = ("rmsnorm", "layernorm"),
    activations: tuple[str, ...] = ("gelu", "relu", "silu"),
    layernorm_eps_range: tuple[float, float] = (1e-6, 1e-4),
    ff_biases: tuple[bool, ...] = (True, False),
    tune_corpus_weights: bool = True,
):
    """
    Optuna Hyperparameter Optimization for quickmt-train.

    Args:
        config: Path to the config file (e.g. configs/faen-tiny.yaml)
        metric: Validation metric to optimize (e.g. loss, bleu, chrf, acc)
        max_steps: Number of steps to run per trial (overrides config train.max_steps)
        eval_steps: Number of steps between validation loops (overrides config train.eval_steps)
        n_trials: Number of Optuna trials to run
        study_name: Name of the experiment study. Defaults to the database file name.
        db: Path to SQLite DB to save Optuna study. Default: optuna_{train.experiment_name}.db
        lr_range: Range for learning rate (min, max)
        weight_decay_range: Range for weight decay (min, max)
        grad_clip_range: Range for gradient clipping (min, max)
        label_smoothing_range: Range for label smoothing (min, max)
        dropout_range: Range for dropout (min, max)
        norm_types: Possible values for normalization type
        activations: Possible values for activation function
        layernorm_eps_range: Range for layernorm epsilon (min, max)
        ff_biases: Possible values for feed-forward bias
        tune_corpus_weights: Whether to tune the weight of each corpus in the config
    """

    # Create a simple args object to match the expected interface in objective()
    class Args:
        pass

    args = Args()
    args.config = config
    args.metric = metric
    args.max_steps = max_steps
    args.eval_steps = eval_steps
    args.n_trials = n_trials
    args.study_name = study_name
    args.db = db
    args.lr_range = lr_range
    args.weight_decay_range = weight_decay_range
    args.grad_clip_range = grad_clip_range
    args.label_smoothing_range = label_smoothing_range
    args.dropout_range = dropout_range
    args.norm_types = norm_types
    args.activations = activations
    args.layernorm_eps_range = layernorm_eps_range
    args.ff_biases = ff_biases
    args.tune_corpus_weights = tune_corpus_weights

    # Load config to resolve default db path
    _, _, train_cfg, _ = load_config(args.config)
    db_path = (
        args.db if args.db is not None else f"optuna_{train_cfg.experiment_name}.db"
    )
    storage = f"sqlite:///{db_path}"

    final_study_name = (
        args.study_name
        if args.study_name is not None
        else os.path.splitext(os.path.basename(db_path))[0]
    )

    direction = "minimize" if args.metric in ["loss", "ppl"] else "maximize"

    # Pruner trims trials that are performing unpromisingly relative to others
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=args.eval_steps * 2
    )
    study = optuna.create_study(
        study_name=final_study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        pruner=pruner,
    )

    print(f"Starting Optuna search. Optimizing {args.metric} ({direction}).")
    print(f"Running {args.n_trials} trials for {args.max_steps} steps each.")

    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n--- Search Finished ---")
    if len(study.trials) > 0 and study.best_trial is not None:
        try:
            print(f"Best trial (Number {study.best_trial.number}):")
            print(f"  Value ({args.metric}): {study.best_trial.value:.4f}")
            print("  Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
        except Exception:
            print("No trials completed.")
    else:
        print("No trials finished successfully.")


def main():
    import fire

    fire.Fire(search_cli)


if __name__ == "__main__":
    main()
