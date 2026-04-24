import torch
import os
import fire
import json
from safetensors.torch import load_file, save_model
from .checkpoint_utils import get_best_steps, extract_step
from .config import load_config
from .model import Seq2SeqTransformer
from .data import PrepareData


def main():
    fire.Fire(average_checkpoints_cli)


def get_averaged_state_dict(experiment_dir, model_cfg, train_cfg, export_cfg):
    """
    Find and average the best k checkpoints.
    """
    metrics_path = os.path.join(experiment_dir, "metrics.jsonl")
    best_steps = get_best_steps(
        metrics_path,
        train_cfg.early_stopping_metric.value,
        train_cfg.early_stopping_metric.lower_is_better,
        export_cfg.k
    )

    if best_steps:
        selected = [f"model_{step}.safetensors" for step in best_steps]
        # Verify files exist
        selected = [
            f
            for f in selected
            if os.path.exists(os.path.join(train_cfg.checkpoint_dir, f))
        ]
        print(f"Selected {len(selected)} best checkpoints based on {train_cfg.early_stopping_metric.value}.")
    else:
        print(
            f"No suitable metrics found in {metrics_path}. Falling back to last k checkpoints."
        )
        if not os.path.exists(train_cfg.checkpoint_dir):
            print(f"Directory {train_cfg.checkpoint_dir} not found.")
            return None

        checkpoints = [
            f
            for f in os.listdir(train_cfg.checkpoint_dir)
            if f.startswith("model_")
            and f.endswith(".safetensors")
            and "_int8" not in f
        ]

        # Sort by step number
        checkpoints.sort(key=extract_step, reverse=True)
        selected = checkpoints[: export_cfg.k]

    if not selected:
        print("No model files found.")
        return None

    print(f"Averaging {len(selected)} model checkpoints:")
    for c in selected:
        print(f" - {c}")

    # 2. Load and average state dicts
    avg_state_dict: dict[str, torch.Tensor] = {}
    count = len(selected)

    for i, ckpt_name in enumerate(selected):
        ckpt_path = os.path.join(train_cfg.checkpoint_dir, ckpt_name)
        clean_state_dict = load_file(ckpt_path, device="cpu")

        if not avg_state_dict:
            avg_state_dict = clean_state_dict
        else:
            for k in clean_state_dict:
                if k in avg_state_dict:
                    avg_state_dict[k] += clean_state_dict[k]
                else:
                    # This might happen if mixing different architectures
                    print(f"Warning: Key {k} not found in first checkpoint. Skipping.")

    # Divide by count
    for k in avg_state_dict:
        # Only divide floating point tensors
        if avg_state_dict[k].is_floating_point():
            avg_state_dict[k] = avg_state_dict[k] / count
        else:
            avg_state_dict[k] = torch.div(
                avg_state_dict[k], count, rounding_mode="floor"
            )

    # 3. Save as .pt and .safetensors (FP32/Averaged weights)
    # Ensure experiment directory exists
    os.makedirs(train_cfg.experiment_name, exist_ok=True)

    pt_output = f"{export_cfg.output_prefix}.pt"
    torch.save({"model_state_dict": avg_state_dict}, pt_output)

    st_output = f"{export_cfg.output_prefix}.safetensors"
    # Create model to handle shared tensors correctly during save
    model = Seq2SeqTransformer(model_cfg).to("cpu")
    model.load_state_dict(avg_state_dict, strict=False)
    save_model(model, st_output)
    print(f"Saved averaged model to {pt_output} and {st_output}")

    return avg_state_dict


def average_checkpoints_cli(experiment_dir: str, **kwargs):
    """
    Average the last k checkpoints and save as safetensors/INT8.

    Args:
        experiment_dir: Path to experiment directory
        **kwargs: Overrides for configuration parameters
    """
    model_cfg, data_cfg, train_cfg, export_cfg = load_config(
        os.path.join(experiment_dir, "config.yaml")
    )

    # Apply overrides
    for key, value in kwargs.items():
        found = False
        for cfg in [model_cfg, data_cfg, train_cfg, export_cfg]:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                found = True
        if not found:
            print(f"Warning: Configuration key '{key}' not found in any config object.")

    get_averaged_state_dict(experiment_dir, model_cfg, train_cfg, export_cfg)


if __name__ == "__main__":
    main()
