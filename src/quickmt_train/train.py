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
from torch.cuda.amp import GradScaler

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

from .config import DataConfig, ModelConfig, TrainConfig
from .data import PrepareData
from .model import Seq2SeqTransformer


def print_model_details(model, model_cfg, data_cfg, train_cfg, get_time_info):
    # Calculate parameters for sub-modules
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

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
        dist.init_process_group("nccl")
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
        run["hparams"] = {
            **{f"model_{k}": v for k, v in dataclasses.asdict(model_cfg).items()},
            **{f"data_{k}": v for k, v in dataclasses.asdict(data_cfg).items()},
            **{f"train_{k}": v for k, v in dataclasses.asdict(train_cfg).items()},
        }

    # Data
    if is_main:
        print(f"{get_time_info()} Preparing data...")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    global_step_value = ctx.Value("i", 0)

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

    model_BA = None
    if getattr(train_cfg, 'dual_learning_alpha', 0.0) > 0.0:
        import copy
        model_cfg_ba = copy.deepcopy(model_cfg)
        model_cfg_ba.vocab_size_src = model_cfg.vocab_size_tgt
        model_cfg_ba.vocab_size_tgt = model_cfg.vocab_size_src
        model_BA = Seq2SeqTransformer(model_cfg_ba).to(device)
        print(f"{get_time_info()} Initialized backward model for dual learning...")

    # Point: model weights remain in FP32 (master weights) for Mixed Precision Training.
    load_model_weights(model, train_cfg, device, get_time_info)

    # NOTE: torch.compile is disabled for multi-GPU DDP runs.
    if (
        world_size == 1
        and train_cfg.enable_torch_compile
        and not (torch.cuda.device_count() > 1 and train_cfg.device in ["cuda", "auto"])
    ):
        if is_main:
            print(
                f"{get_time_info()} Attempting to enable torch.compile for single-GPU training"
            )
        try:
            # When using torch.compile, we can use Inductor's native recomputation
            # which is more efficient and avoids conflicts with manual checkpoint().
            if model_cfg.use_checkpoint and inductor_config is not None:
                inductor_config.recompute_all = True
                # Disable manual checkpointing because we use Inductor native instead
                model.config.use_checkpoint = False

            model = torch.compile(model, dynamic=True)

            if model_cfg.use_checkpoint and inductor_config is not None and is_main:
                print(
                    f"{get_time_info()} Enabled Inductor native recomputation (memory optimization)"
                )
        except Exception as e:
            print(f"{get_time_info()} Failed to enable torch.compile: {e}")
            print(f"{get_time_info()} Falling back to non-compiled mode")

    # Wrap model in DDP/DP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        if model_BA is not None:
            model_BA = DDP(
                model_BA,
                device_ids=[local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
        if is_main:
            print(
                f"{get_time_info()} Using DistributedDataParallel (World Size: {world_size})"
            )
    elif torch.cuda.device_count() > 1 and train_cfg.device in ["cuda", "auto"]:
        print(
            f"{get_time_info()} Detected {torch.cuda.device_count()} GPUs. Using DataParallel."
        )
        model = nn.DataParallel(model)
        if model_BA is not None:
             model_BA = nn.DataParallel(model_BA)

    if is_main:
        print_model_details(model, model_cfg, data_cfg, train_cfg, get_time_info)

    # Separate parameters into two groups: those that will receive weight decay and those that will not.
    # Modern Transformer training (BERT, GPT-2, etc.) excludes biases and normalization/embedding
    # parameters from weight decay to improve stability and avoid underfitting.

    # We want to decay: Weight of Linear layers (2D+)
    # We do NOT want to decay: Bias of anything, Weight of Norm/Embedding layers
    decay_params = []
    no_decay_params = []

    models_list = [model]
    if model_BA is not None:
        models_list.append(model_BA)

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
    global_step = 0

    # Checkpoint loading (state)
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

    for batch_idx, (src, tgt) in enumerate(train_loader):
        # Use non_blocking for async data transfer
        src, tgt = (
            src.to(device, non_blocking=True),
            tgt.to(device, non_blocking=True),
        )

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            if getattr(train_cfg, 'dual_learning_alpha', 0.0) > 0.0:
                raw_m = model.module if hasattr(model, "module") else model
                raw_mba = model_BA.module if hasattr(model_BA, "module") else model_BA

                loss_AB, num_tokens = model(src, tgt, label_smoothing=train_cfg.label_smoothing)
                loss_BA, _ = model_BA(tgt, src, label_smoothing=train_cfg.label_smoothing)
                
                if global_step >= getattr(train_cfg, 'dual_learning_warmup_steps', 0):
                    soft_tgt = raw_m.generate_gumbel(src, max_len=tgt.size(1), tau=1.0)
                    loss_cyc_1, _ = model_BA(src=None, src_probs=soft_tgt, tgt=src, label_smoothing=train_cfg.label_smoothing)
                    
                    soft_src = raw_mba.generate_gumbel(tgt, max_len=src.size(1), tau=1.0)
                    loss_cyc_2, _ = model(src=None, src_probs=soft_src, tgt=tgt, label_smoothing=train_cfg.label_smoothing)
                    
                    loss = loss_AB + loss_BA + train_cfg.dual_learning_alpha * (loss_cyc_1 + loss_cyc_2)
                else:
                    loss = loss_AB + loss_BA

            elif train_cfg.rdrop_alpha > 0.0 and global_step >= getattr(train_cfg, 'rdrop_warmup_steps', 0):
                loss1, (logits1, num_tokens) = model(
                    src, tgt, return_outputs=True, label_smoothing=train_cfg.label_smoothing
                )
                loss2, (logits2, _) = model(
                    src, tgt, return_outputs=True, label_smoothing=train_cfg.label_smoothing
                )

                ce_loss = 0.5 * (loss1 + loss2)

                # Compute KL divergence
                p_loss = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
                q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')

                p_loss = p_loss.sum(dim=-1)
                q_loss = q_loss.sum(dim=-1)

                pad_mask = (tgt[:, 1:] == model_cfg.pad_id)
                p_loss.masked_fill_(pad_mask, 0.0)
                q_loss.masked_fill_(pad_mask, 0.0)

                kl_loss = (p_loss.sum() + q_loss.sum()) / 2

                loss = ce_loss + train_cfg.rdrop_alpha * kl_loss
            else:
                loss, num_tokens = model(
                    src, tgt, label_smoothing=train_cfg.label_smoothing
                )

            # Handle DataParallel output (vectors per GPU)
            if loss.ndim > 0:
                loss = loss.sum()
            if num_tokens.ndim > 0:
                num_tokens = num_tokens.sum()

        # In DDP, backward() syncs gradients across all GPUs.
        # Use no_sync context to only sync at the end of accumulation.
        if world_size > 1 and (batch_idx + 1) % train_cfg.accum_steps != 0:
            context = model.no_sync()
        else:
            from contextlib import nullcontext

            context = nullcontext()

        with context:
            scaler.scale(loss).backward()

        accum_loss += loss.item()
        accum_tokens += num_tokens.item()

        total_loss_sum += loss.item()
        total_tokens_trained += num_tokens.item()

        # Throughput tracking
        batch_src_tokens += (src != model_cfg.pad_id).sum().item()
        batch_tgt_tokens += (tgt != model_cfg.pad_id).sum().item()

        if (batch_idx + 1) % train_cfg.accum_steps == 0:
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
            optimizer.zero_grad(set_to_none=True)

            last_batch_loss = accum_loss / max(1, accum_tokens)
            accum_loss = 0
            accum_tokens = 0
            global_step += 1
            global_step_value.value = global_step

            # Validation and Checkpointing
            # Run validate() on ALL ranks (DDP requires all ranks to participate
            # in the forward pass). Only rank 0 logs metrics and saves checkpoints.
            if global_step % train_cfg.eval_steps == 0:
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
                        )
                if on_eval_step is not None:
                    on_eval_step(val_metrics, global_step)

            # Progress Print (Rank 0 only)
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
                    )
            if on_eval_step is not None:
                on_eval_step(val_metrics, global_step)

    return latest_val_metrics


def save_checkpoint(
    step, model, optimizer, scheduler, config, get_time_info, val_metrics=None
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
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
    save_model(raw_model, path)
    print(f"{get_time_info()} Model weights saved: {path}")

    # Save full state (optimizer, scheduler) in .pt for resuming
    path_pt = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path_pt,
    )
    print(f"{get_time_info()} Training state saved: {path_pt}")

    # If it's a quantized model, also save a converted version for inference
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
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

    # Rotation
    def get_step(f):
        try:
            # model_1000.safetensors or checkpoint_1000.pt
            return int(f.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            return -1

    all_files = os.listdir(config.checkpoint_dir)
    checkpoints_pt = sorted(
        [f for f in all_files if f.startswith("checkpoint_")], key=get_step
    )
    models_st = sorted([f for f in all_files if f.startswith("model_")], key=get_step)

    if len(checkpoints_pt) > config.max_checkpoints:
        os.remove(os.path.join(config.checkpoint_dir, checkpoints_pt[0]))
        print(f"{get_time_info()} Removed old state: {checkpoints_pt[0]}")
    if len(models_st) > config.max_checkpoints:
        os.remove(os.path.join(config.checkpoint_dir, models_st[0]))
        print(f"{get_time_info()} Removed old weights: {models_st[0]}")


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

    # Limit samples for BLEU calculation to reduce memory
    max_samples = train_cfg.val_max_samples
    hypotheses = []
    references = []
    sample_count = 0

    # Use inference_mode instead of no_grad for better performance
    autocast_dtype = torch.float32
    if device.type == "cuda":
        if train_cfg.precision in ("bf16", "bfloat16"):
            autocast_dtype = torch.bfloat16
        elif train_cfg.precision in ("fp16", "float16"):
            autocast_dtype = torch.float16

    with torch.inference_mode():
        for batch_idx, (src, tgt) in enumerate(loader):
            src, tgt = (
                src.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )

            # Forward pass for loss and logits (calculates loss internally)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss_sum, (logits, num_tokens_batch) = model(
                    src, tgt, return_outputs=True
                )

            # Handle DataParallel output (vectors per GPU)
            if loss_sum.ndim > 0:
                loss_sum = loss_sum.sum()
            if num_tokens_batch.ndim > 0:
                num_tokens_batch = num_tokens_batch.sum()

            # Accumulate loss and tokens
            total_loss_sum += loss_sum.item()
            total_tokens += num_tokens_batch.item()

            # Accuracy calculation
            tgt_labels = tgt[:, 1:]
            preds = logits.argmax(dim=-1)
            mask_acc = tgt_labels != model_cfg.pad_id
            correct_tokens += ((preds == tgt_labels) & mask_acc).sum().item()

            # Generation for BLEU/ChrF - only process if we still need samples
            if sample_count < max_samples:
                if use_autoregressive:
                    # True autoregressive generation including encoding
                    raw_model = model.module if hasattr(model, "module") else model
                    enc = raw_model.encode(src)
                    generated_ids = raw_model.generate(
                        src,
                        max_len=model_cfg.max_len,
                        enc_output=enc,
                        bos_id=model_cfg.bos_id,
                        eos_id=model_cfg.eos_id,
                    )
                else:
                    # Teacher-forced predictions (fastest, uses existing logits)
                    generated_ids = preds

                for i in range(src.size(0)):
                    if sample_count >= max_samples:
                        break
                    # Post-process: stop at EOS or PAD tokens
                    ids = generated_ids[i].tolist()
                    # Find first EOS or PAD token and truncate
                    for idx, token_id in enumerate(ids):
                        if token_id == model_cfg.eos_id or token_id == model_cfg.pad_id:
                            ids = ids[:idx]
                            break
                    hyp = tgt_sp.decode(ids)
                    ref = tgt_sp.decode(tgt[i].tolist())
                    hypotheses.append(hyp)
                    references.append(ref)
                    sample_count += 1

    avg_loss = total_loss_sum / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    acc = correct_tokens / max(1, total_tokens)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

    print(
        f"\n{get_time_info()} [Validation] Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}"
    )
    for i in range(min(train_cfg.quick_test_samples, len(hypotheses))):
        print(f"Sample {i}:")
        print(f"  Ref: {references[i]}")
        print(f"  Hyp: {hypotheses[i]}")
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
                raw_model = model.module if hasattr(model, "module") else model
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
        if hasattr(train_cfg, key):
            setattr(train_cfg, key, value)
        elif hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
        elif hasattr(data_cfg, key):
            setattr(data_cfg, key, value)
        else:
            print(f"Warning: Configuration key '{key}' not found in any config object.")

    # Make experiment folder if not exists
    os.makedirs(train_cfg.experiment_name, exist_ok=True)

    # Copy config to experiment folder
    copyfile(config, os.path.join(train_cfg.experiment_name, "config.yaml"))  # type: ignore

    train(model_cfg, data_cfg, train_cfg)


if __name__ == "__main__":
    main()
