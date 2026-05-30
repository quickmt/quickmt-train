import math
import os
import time
import json
from datetime import datetime, timedelta
import itertools

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler

import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024  # Default is 64

try:
    import torch._inductor.config as inductor_config
    import torch._inductor.lowering

    # PyTorch's Inductor compiler patch for dynamic 2D tensors.
    if (
        hasattr(torch.ops.aten.logsumexp, "default")
        and torch.ops.aten.logsumexp.default in torch._inductor.lowering.lowerings
    ):
        torch._inductor.lowering.lowerings[torch.ops.aten.logsumexp.default] = (
            torch._inductor.lowering.make_fallback(torch.ops.aten.logsumexp.default)
        )
except ImportError:
    inductor_config = None

from .tracker import get_tracker
from safetensors.torch import save_model
from shutil import copyfile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import CheckpointStrategy, DataConfig, ModelConfig, TrainConfig, serialize_config
from .checkpoint_utils import EMA, save_checkpoint
from .utils import unwrap_model, print_model_details, setup_dist, load_model_weights
from .evaluator import validate, run_quick_test
from .data import PrepareData
from .model import Seq2SeqTransformer


def train(model_cfg=None, data_cfg=None, train_cfg=None, on_eval_step=None):
    training_start = time.time()

    def get_time_info(return_raw=False):
        elapsed = time.time() - training_start
        if return_raw:
            return elapsed
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        curr_time = datetime.now().strftime("%H:%M:%S")
        return f"[{curr_time}] [{elapsed_str}]"

    # Configs
    if model_cfg is None:
        model_cfg = ModelConfig()
    if data_cfg is None:
        data_cfg = DataConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    metrics_path = os.path.join(train_cfg.experiment_name, "metrics.jsonl")
    if not train_cfg.resume_from and os.path.exists(metrics_path):
        print(f"Error: Metrics file '{metrics_path}' already exists.")
        print("Training loop will not start.")
        print("Please remove the file (and move/backup the checkpoints, because they will be removed by the training loop).")
        import sys
        sys.exit(1)

    rank, local_rank, world_size = setup_dist(train_cfg)
    is_main = rank == 0

    try:
        return _train_impl(
            model_cfg,
            data_cfg,
            train_cfg,
            rank,
            local_rank,
            world_size,
            is_main,
            get_time_info,
            on_eval_step=on_eval_step,
        )
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def _train_impl(
    model_cfg,
    data_cfg,
    train_cfg,
    rank,
    local_rank,
    world_size,
    is_main,
    get_time_info,
    on_eval_step=None,
):

    if train_cfg.device == "auto":
        device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(train_cfg.device)

    # Performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = train_cfg.tf32
        torch.backends.cudnn.allow_tf32 = train_cfg.tf32
        if train_cfg.tf32:
            torch.set_float32_matmul_precision("high")

    # Disable specific Inductor features that cause SymPy AssertionErrors
    # with dynamic shapes in certain PyTorch versions.
    if train_cfg.enable_torch_compile and inductor_config is not None:
        inductor_config.coordinate_descent_tuning = False
        inductor_config.coordinate_descent_check_all_directions = False
        # Disable memory coalescing analysis in Triton tiling
        if hasattr(inductor_config, "triton"):
            inductor_config.triton.coalesce_tiling_analysis = False

        if is_main:
            print(
                f"{get_time_info()} Applied Inductor config patches for stability (including Triton tiling bypass)"
            )

    if is_main:
        # Remove metrics file if exists
        metrics_path = os.path.join(train_cfg.experiment_name, "metrics.jsonl")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

        print(f"{get_time_info()} Rank: {rank}/{world_size} | Local Rank: {local_rank}")
        print(f"{get_time_info()} Using device: {device}")

    if is_main:
        repo = getattr(train_cfg, "tracker_repo", None) or getattr(
            train_cfg, "aim_repo", None
        )
        if repo:
            tracker_type = getattr(train_cfg, "tracker", "aim")
            if hasattr(tracker_type, "value"):
                tracker_type = tracker_type.value
            run = get_tracker(tracker_type, repo, train_cfg.experiment_name)
        else:
            run = None
    else:
        run = None

    if run and is_main:
        hparams = {
            **{
                f"model_{k}": v
                for k, v in serialize_config(model_cfg).items()
            },
            **{
                f"data_{k}": v
                for k, v in serialize_config(data_cfg).items()
            },
            **{
                f"train_{k}": v
                for k, v in serialize_config(train_cfg).items()
            },
        }
        run.log_hparams(hparams)

    # Tokenizer preparation
    from .data import PrepareData

    # We call PrepareData here to ensure tokenizers are trained if they don't exist.
    # Only rank 0 prepares data to avoid race conditions when training tokenizers.
    # All other ranks wait at the barrier until rank 0 finishes, then load
    # the already-existing tokenizers.

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    global_step_value = ctx.Value("i", 0)

    if is_main:
        train_loader, dev_loader, src_sp, tgt_sp = PrepareData(
            model_cfg,
            data_cfg,
            train_cfg,
            global_step_value=global_step_value,
            rank=rank,
            world_size=world_size,
        )

    if world_size > 1:
        dist.barrier()

    if not is_main:
        train_loader, dev_loader, src_sp, tgt_sp = PrepareData(
            model_cfg,
            data_cfg,
            train_cfg,
            global_step_value=global_step_value,
            rank=rank,
            world_size=world_size,
        )

    # Model
    print(f"{get_time_info()} Initializing model...")
    if getattr(model_cfg, "attn_logit_softcap", None) is not None:
        print(f"{get_time_info()} Info: Attention logit softcapping enabled (cap: {model_cfg.attn_logit_softcap})")
    if getattr(model_cfg, "final_logit_softcap", None) is not None:
        print(f"{get_time_info()} Info: Final logit softcapping enabled (cap: {model_cfg.final_logit_softcap})")

    model = Seq2SeqTransformer(model_cfg).to(device)
    initial_dropout = model_cfg.dropout

    # Load model weights if resume_from is specified
    load_model_weights(model, train_cfg, device, get_time_info)

    # torch.compile is applied to the INNER model BEFORE DDP wrapping.
    #
    # Compiling after DDP wraps the DDP all-reduce hooks into the compiled graph,
    # which causes hard-to-diagnose deadlocks: torch.compile may trace the backward
    # graph once (with or without all-reduce depending on require_backward_grad_sync
    # at trace time) and then bake that decision in permanently for all future calls.
    # By compiling the inner model first, DDP's hooks remain outside the compiled
    # graph and fire correctly at runtime based on require_backward_grad_sync.
    if train_cfg.enable_torch_compile:
        if is_main:
            print(f"{get_time_info()} Attempting to enable torch.compile")
        try:
            if model_cfg.checkpoint_gradients and inductor_config is not None:
                inductor_config.recompute_all = True
                model.config.checkpoint_gradients = False
                if is_main:
                    print(
                        f"{get_time_info()} Enabled Inductor native recomputation (memory optimization)"
                    )

            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"{get_time_info()} Failed to enable torch.compile: {e}")
            print(f"{get_time_info()} Falling back to non-compiled mode")

    # Wrap compiled model in DDP/DP.
    # ddp_model holds the DDP reference so we can set require_backward_grad_sync
    # directly on it for gradient accumulation without using no_sync().
    ddp_model = None
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            # static_graph=True, # Do not use static graph if shapes are dynamic
        )
        ddp_model = model
        if is_main:
            print(
                f"{get_time_info()} Using DistributedDataParallel (World Size: {world_size})"
            )
    elif torch.cuda.device_count() > 1 and train_cfg.device in ["cuda", "auto"]:
        print(
            f"{get_time_info()} Detected {torch.cuda.device_count()} GPUs. Using DataParallel."
        )
        model = nn.DataParallel(model)

    if is_main:
        print_model_details(model, model_cfg, data_cfg, train_cfg, get_time_info)

    # Separate parameters into two groups: those that will receive weight decay and those that will not.
    # Biases and normalization/embedding parameters are typically excluded from weight decay.

    # We want to decay: Weight of Linear layers (2D+)
    # We do NOT want to decay: Bias of anything, Weight of Norm/Embedding layers
    decay_params = []
    no_decay_params = []

    # Get the raw model to check module types during parameter traversal
    raw_model = unwrap_model(model)

    # Create a mapping from parameter object to its containing module
    param_to_module = {}
    for m in raw_model.modules():
        for p in m.parameters(recurse=False):
            param_to_module[p] = m

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Determine if this parameter should be excluded from weight decay
        module = param_to_module.get(p)
        is_norm = isinstance(module, (nn.LayerNorm, nn.RMSNorm))

        if pn.endswith("bias") or p.ndim == 1 or is_norm:
            no_decay_params.append(p)
        elif not getattr(train_cfg, "weight_decay_embeddings", True) and "emb" in pn:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_opt = train_cfg.fused_optimizer if device.type == "cuda" else False
    if getattr(train_cfg, "use_8bit_optimizer", False):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=train_cfg.lr,
                eps=train_cfg.adam_eps,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            )
            if is_main:
                print(f"{get_time_info()} Using bitsandbytes 8-bit AdamW optimizer")
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for use_8bit_optimizer=True. Please install it with 'pip install bitsandbytes'"
            )
    else:
        optimizer = optim.AdamW(
            optim_groups,
            lr=train_cfg.lr,
            eps=train_cfg.adam_eps,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            fused=fused_opt,
        )

    # Scheduler
    def lr_lambda(current_step):
        # current_step is the number of scheduler.step() calls made so far (0-indexed)
        # We want to treat the first step as 1
        step = current_step + 1
        if train_cfg.scheduler_type == "cosine":
            if step < train_cfg.warmup_steps:
                return float(step) / float(max(1, train_cfg.warmup_steps))
            progress = float(step - train_cfg.warmup_steps) / float(
                max(1, train_cfg.max_steps - train_cfg.warmup_steps)
            )
            # Cosine decay from peak lr to 0.1 * peak lr
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
            return 0.1 + 0.9 * cosine_decay
        else:
            # Inverse Square Root scheduler
            if step < train_cfg.warmup_steps:
                return float(step) / float(max(1, train_cfg.warmup_steps))
            else:
                # Scale so that at warmup_steps, factor is 1.0
                return (train_cfg.warmup_steps**0.5) * (step**-0.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA Initialization
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None

    global_step = 0

    # Checkpoint loading (state)
    best_val_metric = None
    steps_since_best = 0
    if (
        train_cfg.resume_from
        and train_cfg.resume_from.endswith(".pt")
        and not train_cfg.reset_optimizer
    ):
        checkpoint = torch.load(train_cfg.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            print(
                f"{get_time_info()} Resuming optimizer and scheduler state from {train_cfg.resume_from}"
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = checkpoint.get("step", 0)
            global_step_value.value = global_step
            if ema is not None and "ema_state_dict" in checkpoint:
                print(f"{get_time_info()} Resuming EMA state")
                ema.load_state_dict(checkpoint["ema_state_dict"])
            if (
                checkpoint.get("best_val_metric") is not None
                and train_cfg.early_stopping_patience > 0
            ):
                best_val_metric = checkpoint["best_val_metric"]
            print(f"{get_time_info()} Resumed from step {global_step}")

    # Loop
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Mixed Precision Setup
    autocast_dtype = torch.float32
    use_scaler = False
    if device.type == "cuda":
        if train_cfg.precision in ("bf16", "bfloat16"):
            autocast_dtype = torch.bfloat16
        elif train_cfg.precision in ("fp16", "float16"):
            autocast_dtype = torch.float16
            use_scaler = True

    scaler = GradScaler(enabled=use_scaler)

    start_time = time.time()
    total_loss_sum = 0
    total_tokens_trained = 0
    batch_src_tokens = 0
    batch_tgt_tokens = 0
    last_log_time = time.time()

    # Token-based accumulation state
    accum_loss = 0
    accum_tokens = 0
    last_batch_loss = 0.0
    last_grad_norm = 0.0
    clipping_count = 0  # Count of clipping events since last log
    latest_val_metrics = None

    for batch_idx, (src, tgt) in enumerate(train_loader, start=1):
        # Use non_blocking for async data transfer
        src, tgt = (
            src.to(device, non_blocking=True),
            tgt.to(device, non_blocking=True),
        )

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            loss, num_tokens = model(
                src,
                tgt,
                label_smoothing=train_cfg.label_smoothing,
                z_loss_coeff=train_cfg.z_loss_coeff,
            )

            # Handle DataParallel output (vectors per GPU)
            if loss.ndim > 0:
                loss = loss.sum()
            if num_tokens.ndim > 0:
                num_tokens = num_tokens.sum()

        # Backward pass with optional gradient synchronization.
        # We use require_backward_grad_sync rather than the no_sync() context manager.
        # no_sync() is a context manager that torch.compile can trace into the graph and
        # bake in permanently, making it ignore the flag on subsequent compiled calls.
        # require_backward_grad_sync is a plain attribute assignment that DDP checks at
        # runtime during the backward pass, so it works correctly with torch.compile.
        if ddp_model is not None:
            ddp_model.require_backward_grad_sync = (
                batch_idx % train_cfg.accum_steps == 0
            )
        scaler.scale(loss).backward()

        accum_loss += loss.item()
        accum_tokens += num_tokens.item()

        total_loss_sum += loss.item()
        total_tokens_trained += num_tokens.item()

        # Throughput tracking
        batch_src_tokens += (src != model_cfg.pad_id).sum().item()
        batch_tgt_tokens += (tgt != model_cfg.pad_id).sum().item()

        if batch_idx % train_cfg.accum_steps == 0:
            # Scale and clip
            # Scale gradients by total number of tokens in the accumulation bucket
            # For DDP, we must synchronize the token count across all processes to avoid weight divergence.
            if world_size > 1:
                token_tensor = torch.tensor(
                    [accum_tokens], device=device, dtype=torch.float32
                )
                dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
                global_accum_tokens = token_tensor.item()
            else:
                global_accum_tokens = float(accum_tokens)

            # Important: unscale before clipping or manual gradient manipulation
            scaler.unscale_(optimizer)

            for p in model.parameters():
                if p.grad is not None:
                    # DDP averages gradients across ranks, so we multiply by world_size
                    # before dividing by global token count to get the true token-level average.
                    p.grad.data.mul_(world_size).div_(max(1.0, global_accum_tokens))

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            )
            last_grad_norm = (
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            )
            if last_grad_norm > train_cfg.grad_clip:
                clipping_count += 1

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema_start = getattr(train_cfg, "ema_start_step", 0)
                if global_step == ema_start and is_main:
                    print(
                        f"{get_time_info()} EMA smoothing started at step {global_step} (decay: {train_cfg.ema_decay})"
                    )
                ema.update(step=global_step, start_step=ema_start)
            optimizer.zero_grad(set_to_none=True)

            last_batch_loss = accum_loss / max(1, accum_tokens)
            accum_loss = 0
            accum_tokens = 0
            global_step += 1
            global_step_value.value = global_step

            # Dynamic dropout decay with cosine schedule
            if getattr(train_cfg, "decay_dropout", True) and initial_dropout > 0.0:
                if global_step <= train_cfg.warmup_steps:
                    new_dropout = initial_dropout
                else:
                    progress = float(global_step - train_cfg.warmup_steps) / float(
                        max(1, train_cfg.max_steps - train_cfg.warmup_steps)
                    )
                    progress = min(max(progress, 0.0), 1.0)
                    new_dropout = initial_dropout * 0.5 * (1.0 + math.cos(math.pi * progress))
                
                # Round to the configured resolution to prevent frequent torch.compile recompilations
                resolution = getattr(train_cfg, "dropout_decay_resolution", 0.01)
                if resolution > 0.0:
                    new_dropout = round(new_dropout / resolution) * resolution
                
                unwrap_model(model).set_dropout(new_dropout)

            # Validation and Checkpointing
            # Run validate() on ALL ranks (DDP requires all ranks to participate
            # in the forward pass). Only rank 0 saves checkpoints.
            if global_step % train_cfg.eval_steps == 0:
                if ema is not None:
                    ema.apply_shadow()

                val_metrics = validate(
                    model,
                    dev_loader,
                    src_sp,
                    tgt_sp,
                    device,
                    train_cfg,
                    data_cfg,
                    model_cfg,
                    get_time_info,
                )

                if ema is not None:
                    ema.restore()
                latest_val_metrics = val_metrics
                if is_main:
                    if run:
                        for k, v in val_metrics.items():
                            run.track(
                                v,
                                name=f"val_{k}",
                                step=global_step,
                                context={"subset": "dev"},
                            )
                    if getattr(train_cfg, "save_checkpoints", True):
                        save_checkpoint(
                            global_step,
                            model,
                            optimizer,
                            scheduler,
                            train_cfg,
                            get_time_info,
                            val_metrics=val_metrics,
                            ema=ema,
                        )
                if on_eval_step is not None:
                    on_eval_step(val_metrics, global_step)

                # Early Stopping Logic (synchronized for DDP)
                stop_training = torch.tensor(0, device=device)
                if train_cfg.early_stopping_patience > 0:
                    metric = val_metrics.get(train_cfg.early_stopping_metric.value)
                    if metric is not None:
                        if best_val_metric is None or (
                            metric < best_val_metric
                            if train_cfg.early_stopping_metric.lower_is_better
                            else metric > best_val_metric
                        ):
                            best_val_metric, steps_since_best = metric, 0
                        else:
                            steps_since_best += 1

                        if steps_since_best >= train_cfg.early_stopping_patience:
                            if is_main:
                                print(
                                    f"{get_time_info()} Early stopping at step {global_step} (patience reached)"
                                )
                            stop_training.fill_(1)

                if world_size > 1:
                    dist.all_reduce(stop_training, op=dist.ReduceOp.MAX)
                if stop_training.item() > 0:
                    break

                # Restore train mode after validation
                model.train()
            if is_main and global_step % train_cfg.log_steps == 0:
                curr_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - last_log_time
                curr_dropout = unwrap_model(model).config.dropout

                # Estimate total system throughput: rank 0's count × world_size.
                # Data is sharded evenly so this is a good approximation, and avoids
                # introducing any NCCL collectives inside a rank-0-only branch.
                in_tok_s = batch_src_tokens * world_size / max(1e-6, elapsed)
                out_tok_s = batch_tgt_tokens * world_size / max(1e-6, elapsed)

                print(
                    f"{get_time_info()} Step {global_step}/{train_cfg.max_steps} | Batch {batch_idx} | "
                    f"Loss: {last_batch_loss:.4f} | Grad: {last_grad_norm:.4f} "
                    f"(Clipped {clipping_count}x) | LR: {curr_lr:.6f} | Drop: {curr_dropout:.4f} | "
                    f"In: {in_tok_s:.0f} tok/s | Out: {out_tok_s:.0f} tok/s"
                    + (f" ({world_size} GPUs)" if world_size > 1 else "")
                )

                # Aim tracking
                if run:
                    run.track(
                        last_batch_loss,
                        name="loss",
                        step=global_step,
                        context={"subset": "train"},
                    )
                    run.track(curr_lr, name="lr", step=global_step)
                    run.track(curr_dropout, name="dropout", step=global_step)
                    run.track(last_grad_norm, name="grad_norm", step=global_step)
                    run.track(clipping_count, name="clipping_count", step=global_step)
                    run.track(in_tok_s, name="input_tokens_per_sec", step=global_step)
                    run.track(out_tok_s, name="output_tokens_per_sec", step=global_step)

                # Reset tracking counters
                batch_src_tokens = 0
                batch_tgt_tokens = 0
                clipping_count = 0
                last_log_time = time.time()

            if global_step >= train_cfg.max_steps:
                break

    if is_main:
        avg_loss = total_loss_sum / max(1, total_tokens_trained)
        print(
            f"{get_time_info()} Training Completed | Avg Loss: {avg_loss:.4f} | Total Time: {time.time() - start_time:.2f}s"
        )
        print(f"{get_time_info()} Training complete.")

        run_quick_test(
            model,
            dev_loader,
            src_sp,
            tgt_sp,
            device,
            model_cfg,
            train_cfg,
            get_time_info,
        )

        if ema is not None:
            ema.apply_shadow()

        if global_step % train_cfg.eval_steps != 0:
            val_metrics = validate(
                model,
                dev_loader,
                src_sp,
                tgt_sp,
                device,
                train_cfg,
                data_cfg,
                model_cfg,
                get_time_info,
            )
            latest_val_metrics = val_metrics
            if is_main:
                if run:
                    for k, v in val_metrics.items():
                        run.track(
                            v,
                            name=f"val_{k}",
                            step=global_step,
                            context={"subset": "dev"},
                        )
                if getattr(train_cfg, "save_checkpoints", True):
                    save_checkpoint(
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        train_cfg,
                        get_time_info,
                        val_metrics=val_metrics,
                        ema=ema,
                    )
        else:
            if getattr(train_cfg, "save_checkpoints", True):
                save_checkpoint(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    train_cfg,
                    get_time_info,
                    val_metrics=latest_val_metrics,
                    ema=ema,
                )

        if ema is not None:
            ema.restore()

    if is_main and run:
        run.close()

    return latest_val_metrics


def main():
    import fire

    fire.Fire(train_cli)


def train_cli(config: str, **kwargs):
    """
    Train a Transformer model.

    Args:
        config: Path to config file
        **kwargs: Overrides for configuration parameters (e.g., --max_steps 100)
    """
    from .config import load_config, ExportConfig
    import yaml

    model_cfg, data_cfg, train_cfg, export_cfg = load_config(config)

    # Apply overrides
    for key, value in kwargs.items():
        applied = False
        for cfg in [train_cfg, model_cfg, data_cfg, export_cfg]:
            if hasattr(cfg, key):
                from enum import Enum

                existing_val = getattr(cfg, key)
                if isinstance(existing_val, Enum):
                    try:
                        value = type(existing_val)(value)
                    except ValueError:
                        valid_vals = [e.value for e in type(existing_val)]
                        raise ValueError(
                            f"Invalid value '{value}' for {key}. Expected one of {valid_vals}"
                        )
                setattr(cfg, key, value)
                applied = True
                break
        if not applied:
            print(f"Warning: Configuration key '{key}' not found in any config object.")

    # Make experiment folder if not exists
    os.makedirs(train_cfg.experiment_name, exist_ok=True)

    complete_config = {
        "model": serialize_config(model_cfg),
        "data": serialize_config(data_cfg),
        "train": serialize_config(train_cfg),
        "export": serialize_config(export_cfg),
    }

    with open(os.path.join(train_cfg.experiment_name, "config.yaml"), "w") as f:
        yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

    train(model_cfg, data_cfg, train_cfg)


if __name__ == "__main__":
    main()
