import os
import json
import re
import itertools
import torch
from safetensors.torch import save_model

from .utils import unwrap_model
from .config import CheckpointStrategy


def extract_step(filename):
    """
    Extracts the step number from a filename like model_1000.safetensors or checkpoint_1000.pt
    """
    match = re.search(r"_(\d+)(?:_ema)?\.", filename)
    if match:
        return int(match.group(1))
    return -1


def get_best_steps(metrics_path, metric_name, lower_is_better, k=None):
    """
    Returns the step numbers for the top k checkpoints based on the provided metrics.
    If k is None, returns all scored steps sorted by metric.
    """
    if not os.path.exists(metrics_path):
        return []

    scored_steps = []
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                step = entry.get("step")
                metric_val = entry.get(metric_name)
                if step is not None and metric_val is not None:
                    scored_steps.append((metric_val, step))
            except json.JSONDecodeError:
                continue

    # Sort: Primary key is the metric value, secondary key is step (favoring later steps)
    scored_steps.sort(key=lambda x: (x[0] if lower_is_better else -x[0], -x[1]))

    steps = [s for _, s in scored_steps]
    if k is not None:
        return steps[:k]
    return steps


class EMA:
    """
    Exponential Moving Average of model parameters.
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Use the underlying model if it's wrapped in DDP or compiled
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in itertools.chain(
            raw_model.named_parameters(), raw_model.named_buffers()
        ):
            if param.requires_grad or name in self.shadow:
                self.shadow[name] = param.data.clone()

    def update(self, step=None, start_step=0):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in itertools.chain(
            raw_model.named_parameters(), raw_model.named_buffers()
        ):
            if param.requires_grad or name in self.shadow:
                assert name in self.shadow
                if step is not None and step < start_step:
                    # Sync exactly with model weights before the start step
                    self.shadow[name].copy_(param.data)
                else:
                    # shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].copy_(
                        self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    )

    def apply_shadow(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in itertools.chain(
            raw_model.named_parameters(), raw_model.named_buffers()
        ):
            if param.requires_grad or name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in itertools.chain(
            raw_model.named_parameters(), raw_model.named_buffers()
        ):
            if param.requires_grad or name in self.shadow:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, shadow_param in self.shadow.items():
            if name in state_dict:
                shadow_param.copy_(state_dict[name])


def save_checkpoint(
    step, model, optimizer, scheduler, config, get_time_info, val_metrics=None, ema=None
):
    # Free up memory before potentially expensive save operations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ensure experiment directory exists
    os.makedirs(config.experiment_name, exist_ok=True)

    # Save validation metrics to jsonl
    if val_metrics is not None:
        metrics_path = os.path.join(config.experiment_name, "metrics.jsonl")
        with open(metrics_path, "a") as f:
            elapsed = get_time_info(return_raw=True)
            metric_entry = {"step": step, "elapsed": round(elapsed, 2), **val_metrics}
            f.write(json.dumps(metric_entry) + "\n")

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # Use save_model instead of save_file to handle shared tensors (tied embeddings)
    # We need to unwrap the model to get the underlying structure for save_model
    raw_model = unwrap_model(model)

    path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
    save_model(raw_model, path)
    print(f"{get_time_info()} Model weights saved: {path}")

    if ema is not None and step >= getattr(config, "ema_start_step", 0):
        ema_path = os.path.join(config.checkpoint_dir, f"model_{step}_ema.safetensors")
        # Temporarily apply EMA weights to save them
        ema.apply_shadow()
        save_model(raw_model, ema_path)
        ema.restore()
        print(f"{get_time_info()} EMA weights saved: {ema_path}")

    # Save full state (optimizer, scheduler) in .pt for resuming
    path_pt = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "best_val_metric": (
                val_metrics.get(config.early_stopping_metric.value)
                if val_metrics
                else None
            ),
        },
        path_pt,
    )
    print(f"{get_time_info()} Training state saved: {path_pt}")

    # If it's a quantized model, also save a converted version for inference
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "qconfig") and raw_model.qconfig is not None:
        import copy

        try:
            quant_model = copy.deepcopy(raw_model)
            quant_model.convert_to_int8()
            quant_path = os.path.join(config.checkpoint_dir, f"model_{step}_int8.pt")
            torch.save(quant_model.state_dict(), quant_path)
            print(f"{get_time_info()} Exported INT8 model: {quant_path}")
        except Exception as e:
            print(f"{get_time_info()} Could not export INT8 model: {e}")

    all_files = os.listdir(config.checkpoint_dir)
    checkpoints_pt = sorted(
        [f for f in all_files if f.startswith("checkpoint_")], key=extract_step
    )
    models_st = sorted(
        [
            f
            for f in all_files
            if f.startswith("model_") and f.endswith(".safetensors") and "_ema" not in f
        ],
        key=extract_step,
    )
    emas_st = sorted(
        [
            f
            for f in all_files
            if f.startswith("model_") and f.endswith("_ema.safetensors")
        ],
        key=extract_step,
    )

    if config.checkpoint_strategy == CheckpointStrategy.BEST:
        metrics_path = os.path.join(config.experiment_name, "metrics.jsonl")
        best_steps = get_best_steps(
            metrics_path,
            config.early_stopping_metric.value,
            config.early_stopping_metric.lower_is_better,
            config.max_checkpoints,
        )

        keep_steps = set(best_steps)
        keep_steps.add(step)  # Always keep current for safety

        if not best_steps:
            print(
                f"{get_time_info()} Warning: no metric scores found, skipping best-checkpoint cleanup"
            )
        else:
            for ckpt_file in checkpoints_pt:
                if extract_step(ckpt_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, ckpt_file))
                    print(
                        f"{get_time_info()} Removed old state (not in top-{config.max_checkpoints}): {ckpt_file}"
                    )

            for model_file in models_st:
                if extract_step(model_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, model_file))
                    print(
                        f"{get_time_info()} Removed old weights (not in top-{config.max_checkpoints}): {model_file}"
                    )

            for ema_file in emas_st:
                if extract_step(ema_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, ema_file))
                    print(
                        f"{get_time_info()} Removed old EMA weights (not in top-{config.max_checkpoints}): {ema_file}"
                    )

    else:
        # Sort checkpoints by step number (recent first) to maintain the most recent checkpoints
        checkpoints_pt = sorted(checkpoints_pt, key=extract_step, reverse=True)
        models_st = sorted(models_st, key=extract_step, reverse=True)
        emas_st = sorted(emas_st, key=extract_step, reverse=True)

        if len(checkpoints_pt) > config.max_checkpoints:
            for ckpt_file in checkpoints_pt[config.max_checkpoints:]:
                os.remove(os.path.join(config.checkpoint_dir, ckpt_file))
                print(f"{get_time_info()} Removed old state: {ckpt_file}")
        if len(models_st) > config.max_checkpoints:
            for model_file in models_st[config.max_checkpoints:]:
                os.remove(os.path.join(config.checkpoint_dir, model_file))
                print(f"{get_time_info()} Removed old weights: {model_file}")
        if len(emas_st) > config.max_checkpoints:
            for ema_file in emas_st[config.max_checkpoints:]:
                os.remove(os.path.join(config.checkpoint_dir, ema_file))
                print(f"{get_time_info()} Removed old EMA weights: {ema_file}")
