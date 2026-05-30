import os
import io
import random
import gzip
import lzma
import zipfile
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import load_file

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def smart_open(filename, mode="r", encoding="utf-8"):
    """
    Open a file with support for gzip, xz, zstd, and zip compression.
    """
    if "b" not in mode:
        if mode == "r":
            mode = "rt"
        elif mode == "w":
            mode = "wt"
        elif mode == "a":
            mode = "at"

    if filename.endswith(".gz"):
        return gzip.open(filename, mode, encoding=encoding if "b" not in mode else None)
    elif filename.endswith(".xz"):
        return lzma.open(filename, mode, encoding=encoding if "b" not in mode else None)
    elif filename.endswith(".zst"):
        if zstd is None:
            raise ImportError(
                "zstandard is not installed. Please install it to open .zst files."
            )
        if "w" in mode or "a" in mode:
            # Use all available cores for compression
            cctx = zstd.ZstdCompressor(threads=-1)
            return zstd.open(
                filename, mode, cctx=cctx, encoding=encoding if "b" not in mode else None
            )
        return zstd.open(filename, mode, encoding=encoding if "b" not in mode else None)
    elif filename.endswith(".zip"):
        if "w" in mode or "a" in mode:
            # Writing to zip is more complex; falling back to standard open for now.
            return open(filename, mode, encoding=encoding)
        zf = zipfile.ZipFile(filename, "r")
        name = zf.namelist()[0]
        return io.TextIOWrapper(zf.open(name), encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)


def create_sample_file(input_files, output_file, target_total_lines):
    """
    Creates a sampled plain text file from one or more input files (potentially compressed).
    SentencePiece training requires plain text files.
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    lines_per_file = max(1, target_total_lines // len(input_files))

    with open(output_file, "w", encoding="utf-8") as f_out:
        for f_path in input_files:
            if not os.path.exists(f_path):
                continue
            with smart_open(f_path, "r", encoding="utf-8") as f_in:
                # Reservoir sampling for random sample without loading full file
                reservoir = []
                for i, line in enumerate(f_in):
                    if i < lines_per_file:
                        reservoir.append(line)
                    else:
                        j = random.randint(0, i)
                        if j < lines_per_file:
                            reservoir[j] = line

                for line in reservoir:
                    f_out.write(line)


def unwrap_model(model):
    """Unwrap a model through torch.compile (_orig_mod) and DDP/DP layers."""
    m = model
    # Peel off compile wrapper
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    # Peel off DDP/DataParallel wrapper
    if hasattr(m, "module"):
        m = m.module
    # One more compile wrapper in case compile was applied before DDP
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def print_model_details(model, model_cfg, data_cfg, train_cfg, get_time_info):
    """Print model trainable parameters and overall configuration info."""
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
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single process
        return 0, 0, 1


def load_model_weights(model, train_cfg, device, get_time_info):
    """Load model weights from a safetensors or PyTorch checkpoint file."""
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
