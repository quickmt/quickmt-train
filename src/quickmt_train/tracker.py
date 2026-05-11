from typing import Dict, Any, Optional

class ExperimentTracker:
    """Base class for experiment trackers."""
    def __init__(self, repo: str, experiment_name: str):
        self.repo = repo
        self.experiment_name = experiment_name

    def log_hparams(self, hparams: Dict[str, Any]):
        pass

    def track(self, value: float, name: str, step: int, context: Optional[Dict[str, str]] = None):
        pass

    def close(self):
        pass

class AimTracker(ExperimentTracker):
    def __init__(self, repo: str, experiment_name: str):
        super().__init__(repo, experiment_name)
        from aim import Run
        self.run = Run(repo=repo, experiment=experiment_name)

    def log_hparams(self, hparams: Dict[str, Any]):
        self.run["hparams"] = hparams

    def track(self, value: float, name: str, step: int, context: Optional[Dict[str, str]] = None):
        if context:
            self.run.track(value, name=name, step=step, context=context)
        else:
            self.run.track(value, name=name, step=step)

    def close(self):
        self.run.close()

class WandbLikeTracker(ExperimentTracker):
    """Common implementation for trackers with a wandb-like API."""
    def log_hparams(self, hparams: Dict[str, Any]):
        self.run.config.update(hparams)

    def track(self, value: float, name: str, step: int, context: Optional[Dict[str, str]] = None):
        metric_name = name
        if context and "subset" in context:
            metric_name = f"{context['subset']}/{name}"
        self.run.log({metric_name: value}, step=step)

    def close(self):
        self.run.finish()

class TrackioTracker(WandbLikeTracker):
    def __init__(self, repo: str, experiment_name: str):
        super().__init__(repo, experiment_name)
        import trackio
        import os
        os.environ["TRACKIO_DIR"] = repo
        self.run = trackio.init(project=experiment_name)

class MLFlowTracker(ExperimentTracker):
    def __init__(self, repo: str, experiment_name: str):
        super().__init__(repo, experiment_name)
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(repo)
        self.mlflow.set_experiment(experiment_name)
        self.run = self.mlflow.start_run()

    def log_hparams(self, hparams: Dict[str, Any]):
        def flatten(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        self.mlflow.log_params(flatten(hparams))

    def track(self, value: float, name: str, step: int, context: Optional[Dict[str, str]] = None):
        metric_name = name
        if context and "subset" in context:
            metric_name = f"{context['subset']}_{name}"
        self.mlflow.log_metric(metric_name, value, step=step)

    def close(self):
        self.mlflow.end_run()

class WandbTracker(WandbLikeTracker):
    def __init__(self, repo: str, experiment_name: str):
        super().__init__(repo, experiment_name)
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(project=experiment_name, dir=repo)

def get_tracker(tracker_type: str, repo: str, experiment_name: str) -> Optional[ExperimentTracker]:
    try:
        if tracker_type == "aim":
            return AimTracker(repo, experiment_name)
        elif tracker_type == "trackio":
            return TrackioTracker(repo, experiment_name)
        elif tracker_type == "mlflow":
            return MLFlowTracker(repo, experiment_name)
        elif tracker_type == "wandb":
            return WandbTracker(repo, experiment_name)
    except ImportError as e:
        print(f"Warning: Failed to import tracker '{tracker_type}': {e}")
        return None
    return None
