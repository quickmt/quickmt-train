import math
import os
import time
import json
from datetime import datetime, timedelta

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler

try:
    import torch._inductor.config as inductor_config
except ImportError:
    inductor_config = None

try:
    from aim import Run

    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None
from safetensors.torch import load_file, save_model
from shutil import copyfile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import CheckpointStrategy, DataConfig, EarlyStoppingMetric, ModelConfig, TrainConfig
from .checkpoint_utils import get_best_steps, extract_step
from .data import PrepareData
from .model import Seq2SeqTransformer


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
            
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, step=None, start_step=0):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in raw_model.named_parameters():
            if param.requires_grad:
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

        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, shadow_param in self.shadow.items():
            if name in state_dict:
                shadow_param.copy_(state_dict[name])


def print_model_details(model, model_cfg, data_cfg, train_cfg, get_time_info):
    # Calculate parameters for sub-modules
    raw_model = unwrap_model(model)

    src_emb_params = sum(p.numel() for p in raw_model.src_tok_emb.parameters())
    tgt_emb_params = sum(p.numel() for p in raw_model.tgt_tok_emb.parameters())
    enc_params = sum(p.numel() for p in raw_model.encoder.parameters())
    dec_params = sum(p.numel() for p in raw_model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{get_time_info()} Model parameters: {total_params:,}")
    print(f"{get_time_info()} Trainable parameters: {trainable_params:,}")
    print(f"{get_time_info()} Source embedding params: {src_emb_params:,}")
    print(f"{get_time_info()} Target embedding params: {tgt_emb_params:,}")
    print(f"{get_time_info()} Encoder params: {enc_params:,}")
    print(f"{get_time_info()} Decoder params: {dec_params:,}")

    # Print model architecture
    print(f"\n{get_time_info()} Model Architecture:")
    print("-" * 60)
    print(model)
    print("-" * 60)

    # Print configs
    print(f"\n{get_time_info()} Configuration:")
    print("-" * 60)

    print("Model Config:")
    for key, value in model_cfg.__dict__.items():
        print(f"  {key}: {value}")

    print("\nData Config:")
    for key, value in data_cfg.__dict__.items():
        print(f"  {key}: {value}")

    print("\nTrain Config:")
    for key, value in train_cfg.__dict__.items():
        print(f"  {key}: {value}")

    print("-" * 60)


def setup_dist(train_cfg):
    """
    Initialize distributed training environment.
    Returns (rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Set a 10-minute NCCL timeout so collective deadlocks surface as a clear
        # error with a stack trace rather than hanging indefinitely.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single process
        return 0, 0, 1


def load_model_weights(model, train_cfg, device, get_time_info):
    if not train_cfg.resume_from:
        return

    checkpoint_path = train_cfg.resume_from
    weights_path = None

    if checkpoint_path.endswith(".safetensors"):
        weights_path = checkpoint_path
    elif checkpoint_path.endswith(".pt"):
        # Could be a full checkpoint or just weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" not in checkpoint:
            # Likely just weights in .pt
            model.load_state_dict(checkpoint)
            print(f"{get_time_info()} Loaded weights from {checkpoint_path}")
        elif isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            # Full checkpoint state, find weights
            step = checkpoint.get("step", 0)
            # Try to find model_{step}.safetensors in the same directory
            weights_path = os.path.join(
                os.path.dirname(checkpoint_path), f"model_{step}.safetensors"
            )
            if not os.path.exists(weights_path):
                print(
                    f"{get_time_info()} Warning: weights file not found for checkpoint at {weights_path}"
                )
                weights_path = None

    if weights_path:
        print(f"{get_time_info()} Loading weights from {weights_path}")
        state_dict = load_file(weights_path, device=device.type)
        # Remove _orig_mod. prefix if present in state_dict (shouldn't be, but safe)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)


def train(model_cfg=None, data_cfg=None, train_cfg=None, on_eval_step=None):
    training_start = time.time()

    def get_time_info():
        elapsed = time.time() - training_start
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
    model_cfg, data_cfg, train_cfg, rank, local_rank, world_size, is_main, get_time_info, on_eval_step=None
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

    if is_main:
        # Remove metrics file if exists
        metrics_path = os.path.join(train_cfg.experiment_name, "metrics.jsonl")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

        print(f"{get_time_info()} Rank: {rank}/{world_size} | Local Rank: {local_rank}")
        print(f"{get_time_info()} Using device: {device}")

    if is_main and AIM_AVAILABLE and train_cfg.aim_repo:
        run = Run(repo=train_cfg.aim_repo, experiment=train_cfg.experiment_name)
    else:
        run = None

    import dataclasses

    if run and is_main:
        from enum import Enum
        def serialize_val(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, list):
                return [serialize_val(i) for i in v]
            if isinstance(v, dict):
                return {k: serialize_val(i) for k, i in v.items()}
            return v

        run["hparams"] = {
            **{f"model_{k}": serialize_val(v) for k, v in dataclasses.asdict(model_cfg).items()},
            **{f"data_{k}": serialize_val(v) for k, v in dataclasses.asdict(data_cfg).items()},
            **{f"train_{k}": serialize_val(v) for k, v in dataclasses.asdict(train_cfg).items()},
        }

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

    model = Seq2SeqTransformer(model_cfg).to(device)

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
            if model_cfg.use_checkpoint and inductor_config is not None:
                inductor_config.recompute_all = True
                model.config.use_checkpoint = False
                if is_main:
                    print(f"{get_time_info()} Enabled Inductor native recomputation (memory optimization)")

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

    models_list = [model]

    for current_model in models_list:
        for pn, p in current_model.named_parameters():
            if not p.requires_grad:
                continue
            if pn.endswith("bias") or p.ndim == 1:
                no_decay_params.append(p)
            elif not getattr(train_cfg, "weight_decay_embeddings", True) and "emb" in pn:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        optim_groups,
        lr=train_cfg.lr,
        eps=train_cfg.adam_eps,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        fused=True if device.type == "cuda" else False,
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
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
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
            if checkpoint.get("best_val_metric") is not None and train_cfg.early_stopping_patience > 0:
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
    clipping_count = 0 # Count of clipping events since last log
    latest_val_metrics = None

    for batch_idx, (src, tgt) in enumerate(train_loader, start=1):
        # Use non_blocking for async data transfer
        src, tgt = (
            src.to(device, non_blocking=True),
            tgt.to(device, non_blocking=True),
        )

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            loss, num_tokens = model(
                src, tgt, label_smoothing=train_cfg.label_smoothing, z_loss_coeff=train_cfg.z_loss_coeff
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
            ddp_model.require_backward_grad_sync = (batch_idx % train_cfg.accum_steps == 0)
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
                    print(f"{get_time_info()} EMA smoothing started at step {global_step} (decay: {train_cfg.ema_decay})")
                ema.update(step=global_step, start_step=ema_start)
            optimizer.zero_grad(set_to_none=True)

            last_batch_loss = accum_loss / max(1, accum_tokens)
            accum_loss = 0
            accum_tokens = 0
            global_step += 1
            global_step_value.value = global_step

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
                        if best_val_metric is None or (metric < best_val_metric if train_cfg.early_stopping_metric.lower_is_better else metric > best_val_metric):
                            best_val_metric, steps_since_best = metric, 0
                        else:
                            steps_since_best += 1
                        
                        if steps_since_best >= train_cfg.early_stopping_patience:
                            if is_main: print(f"{get_time_info()} Early stopping at step {global_step} (patience reached)")
                            stop_training.fill_(1)

                if world_size > 1: dist.all_reduce(stop_training, op=dist.ReduceOp.MAX)
                if stop_training.item() > 0: break
            if is_main and global_step % train_cfg.log_steps == 0:
                curr_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - last_log_time

                # Estimate total system throughput: rank 0's count × world_size.
                # Data is sharded evenly so this is a good approximation, and avoids
                # introducing any NCCL collectives inside a rank-0-only branch.
                in_tok_s = batch_src_tokens * world_size / max(1e-6, elapsed)
                out_tok_s = batch_tgt_tokens * world_size / max(1e-6, elapsed)

                print(
                    f"{get_time_info()} Step {global_step}/{train_cfg.max_steps} | Batch {batch_idx} | "
                    f"Loss: {last_batch_loss:.4f} | Grad: {last_grad_norm:.4f} "
                    f"(Clipped {clipping_count}x) | LR: {curr_lr:.6f} | "
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
            if getattr(train_cfg, "save_checkpoints", True):
                save_checkpoint(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    train_cfg,
                    get_time_info,
                    val_metrics=val_metrics,
                )
        if on_eval_step is not None:
            on_eval_step(val_metrics, global_step)

        if ema is not None:
            ema.restore()

    return latest_val_metrics


def save_checkpoint(
    step, model, optimizer, scheduler, config, get_time_info, val_metrics=None, ema=None
):
    # Ensure experiment directory exists
    os.makedirs(config.experiment_name, exist_ok=True)

    # Save validation metrics to jsonl
    if val_metrics is not None:
        metrics_path = os.path.join(config.experiment_name, "metrics.jsonl")
        with open(metrics_path, "a") as f:
            metric_entry = {"step": step, **val_metrics}
            f.write(json.dumps(metric_entry) + "\n")

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # Use save_model instead of save_file to handle shared tensors (tied embeddings)
    # We need to unwrap the model to get the underlying structure for save_model
    raw_model = unwrap_model(model)

    path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
    save_model(raw_model, path)
    print(f"{get_time_info()} Model weights saved: {path}")

    if ema is not None and step >= getattr(config, 'ema_start_step', 0):
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
            "best_val_metric": val_metrics.get(config.early_stopping_metric.value) if val_metrics else None,
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
    models_st = sorted([f for f in all_files if f.startswith("model_") and f.endswith(".safetensors") and "_ema" not in f], key=extract_step)
    emas_st = sorted([f for f in all_files if f.startswith("model_") and f.endswith("_ema.safetensors")], key=extract_step)

    if config.checkpoint_strategy == CheckpointStrategy.BEST:
        metrics_path = os.path.join(config.experiment_name, "metrics.jsonl")
        best_steps = get_best_steps(
            metrics_path,
            config.early_stopping_metric.value,
            config.early_stopping_metric.lower_is_better,
            config.max_checkpoints
        )
        
        keep_steps = set(best_steps)
        keep_steps.add(step) # Always keep current for safety

        if not best_steps:
            print(f"{get_time_info()} Warning: no metric scores found, skipping best-checkpoint cleanup")
        else:
            for ckpt_file in checkpoints_pt:
                if extract_step(ckpt_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, ckpt_file))
                    print(f"{get_time_info()} Removed old state (not in top-{config.max_checkpoints}): {ckpt_file}")

            for model_file in models_st:
                if extract_step(model_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, model_file))
                    print(f"{get_time_info()} Removed old weights (not in top-{config.max_checkpoints}): {model_file}")
            
            for ema_file in emas_st:
                if extract_step(ema_file) not in keep_steps:
                    os.remove(os.path.join(config.checkpoint_dir, ema_file))
                    print(f"{get_time_info()} Removed old EMA weights (not in top-{config.max_checkpoints}): {ema_file}")
            

    else:
        if len(checkpoints_pt) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, checkpoints_pt[0]))
            print(f"{get_time_info()} Removed old state: {checkpoints_pt[0]}")
        if len(models_st) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, models_st[0]))
            print(f"{get_time_info()} Removed old weights: {models_st[0]}")
        if len(emas_st) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, emas_st[0]))
            print(f"{get_time_info()} Removed old EMA weights: {emas_st[0]}")



def validate(
    model,
    loader,
    src_sp,
    tgt_sp,
    device,
    train_cfg,
    data_cfg,
    model_cfg,
    get_time_info,
    use_autoregressive=True,
):
    """
    Validate the model.
    """
    model.eval()
    total_loss_sum = 0
    total_tokens = 0
    correct_tokens = 0

    hypotheses = []
    references = []

    autocast_dtype = torch.float32
    if device.type == "cuda":
        if train_cfg.precision in ("bf16", "bfloat16"):
            autocast_dtype = torch.bfloat16
        elif train_cfg.precision in ("fp16", "float16"):
            autocast_dtype = torch.float16

    with torch.inference_mode():
        for batch_idx, (src, tgt) in enumerate(loader, start=1):
            src, tgt = (
                src.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss_sum, (logits, num_tokens_batch) = model(
                    src, tgt, return_outputs=True
                )

            if loss_sum.ndim > 0:
                loss_sum = loss_sum.sum()
            if num_tokens_batch.ndim > 0:
                num_tokens_batch = num_tokens_batch.sum()

            total_loss_sum += loss_sum.item()
            total_tokens += num_tokens_batch.item()

            tgt_labels = tgt[:, 1:]
            preds = logits.argmax(dim=-1)
            mask_acc = tgt_labels != model_cfg.pad_id
            correct_tokens += ((preds == tgt_labels) & mask_acc).sum().item()

            if use_autoregressive:
                raw_model = unwrap_model(model)
                enc = raw_model.encode(src)
                generated_ids = raw_model.generate(
                    src,
                    max_len=model_cfg.max_len,
                    enc_output=enc,
                    bos_id=model_cfg.bos_id,
                    eos_id=model_cfg.eos_id,
                )
            else:
                generated_ids = preds

            for i in range(src.size(0)):
                # Stop at EOS or PAD tokens
                def cleanup_ids(ids_list, pad_id, eos_id):
                    for idx, token_id in enumerate(ids_list):
                        if token_id == eos_id or token_id == pad_id:
                            return ids_list[:idx]
                    return ids_list

                ids = cleanup_ids(generated_ids[i].tolist(), model_cfg.pad_id, model_cfg.eos_id)
                ref_ids = cleanup_ids(tgt[i].tolist(), model_cfg.pad_id, model_cfg.eos_id)

                hyp = tgt_sp.decode(ids)
                ref = tgt_sp.decode(ref_ids)
                hypotheses.append(hyp)
                references.append(ref)

    if dist.is_initialized():
        sync_t = torch.tensor([total_loss_sum, float(total_tokens), float(correct_tokens)], device=device)
        dist.all_reduce(sync_t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_tokens, correct_tokens = sync_t.tolist()
        
        all_h, all_r = [None] * dist.get_world_size(), [None] * dist.get_world_size()
        dist.all_gather_object(all_h, hypotheses)
        dist.all_gather_object(all_r, references)
        hypotheses, references = [i for s in all_h for i in s], [i for s in all_r for i in s]

    avg_loss = total_loss_sum / max(1, total_tokens)
    ppl, acc = math.exp(min(avg_loss, 100)), correct_tokens / max(1, total_tokens)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"\n{get_time_info()} [Validation] Loss: {avg_loss:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}")
        for i in range(min(train_cfg.quick_test_samples, len(hypotheses))):
            print(f"Sample {i}: Ref: {references[i][:100]}... | Hyp: {hypotheses[i][:100]}...")
        print("-" * 30)

    model.train()
    return metrics


def run_quick_test(
    model, loader, src_sp, tgt_sp, device, model_cfg, train_cfg, get_time_info
):
    # Quick Test with examples from dev data
    print(
        f"\n{get_time_info()} Running final quick test on {train_cfg.quick_test_samples} dev samples:"
    )
    model.eval()

    samples_found = 0
    with torch.inference_mode():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            # Process up to n samples from this batch
            n = min(train_cfg.quick_test_samples - samples_found, src.size(0))

            for i in range(n):
                s_tensor = src[i : i + 1]
                t_tensor = tgt[i : i + 1]

                # Generate
                raw_model = unwrap_model(model)
                generated_ids = raw_model.generate(
                    s_tensor,
                    max_len=model_cfg.max_len,
                    bos_id=model_cfg.bos_id,
                    eos_id=model_cfg.eos_id,
                )

                # Decoding
                # Helper to remove padding and decode
                def cleanup_and_decode(ids_tensor, sp, pad_id, eos_id):
                    ids = ids_tensor[0].tolist()
                    # Stop at EOS or PAD tokens
                    for idx, token_id in enumerate(ids):
                        if token_id == eos_id or token_id == pad_id:
                            ids = ids[:idx]
                            break
                    return sp.decode(ids)

                s_text = cleanup_and_decode(
                    s_tensor, src_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_ref = cleanup_and_decode(
                    t_tensor, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_hyp = cleanup_and_decode(
                    generated_ids, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )

                print(f"Example {samples_found + 1}:")
                print(f"  Input:  {s_text}")
                print(f"  Ref:    {t_ref}")
                print(f"  Output: {t_hyp}")
                print()

                samples_found += 1

            if samples_found >= train_cfg.quick_test_samples:
                break


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
    from .config import load_config

    model_cfg, data_cfg, train_cfg, _ = load_config(config)

    # Apply overrides
    for key, value in kwargs.items():
        applied = False
        for cfg in [train_cfg, model_cfg, data_cfg]:
            if hasattr(cfg, key):
                from enum import Enum
                existing_val = getattr(cfg, key)
                if isinstance(existing_val, Enum):
                    try:
                        value = type(existing_val)(value)
                    except ValueError:
                        valid_vals = [e.value for e in type(existing_val)]
                        raise ValueError(f"Invalid value '{value}' for {key}. Expected one of {valid_vals}")
                setattr(cfg, key, value)
                applied = True
                break
        if not applied:
            print(f"Warning: Configuration key '{key}' not found in any config object.")

    # Make experiment folder if not exists
    os.makedirs(train_cfg.experiment_name, exist_ok=True)

    # Copy config to experiment folder
    copyfile(config, os.path.join(train_cfg.experiment_name, "config.yaml"))  # type: ignore

    train(model_cfg, data_cfg, train_cfg)


if __name__ == "__main__":
    main()
