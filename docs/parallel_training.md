# Parallel and Multi-Node Training

**NOTE: Work in progress! Multi-GPU training has not yet been tested.**

`quickmt-train` is designed to scale efficiently from a single GPU to multi-node clusters using PyTorch's Distributed Data Parallel (DDP). This guide covers how to set up and optimize parallel training.

## Overview of Parallelism in QuickMT

QuickMT uses **Data Parallelism**:
- The model is replicated on each GPU.
- The training data is sharded across all GPUs (and all nodes).
- Gradients are synchronized and averaged across all processes before each optimizer step.
- Validation metrics (BLEU, ChrF, Loss) are aggregated globally across all shards.

## Single-Node Multi-GPU Training

The easiest way to use multiple GPUs on a single machine is via `torchrun`.

```bash
# Train using all available GPUs on one node
torchrun --nproc_per_node=gpu -m quickmt_train.train configs/your_config.yaml
```

- `--nproc_per_node=gpu`: Automatically detects and uses all available GPUs.
- You can also specify a number, e.g., `--nproc_per_node=4`.
- If you only want to use certain GPUs, set the environmnt variable `CUDA_VISIBLE_DEVICES`, E.G. `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Multi-Node Training

To train across multiple machines, you must ensure:
1. All nodes can communicate via a high-speed network (e.g., InfiniBand).
2. All nodes have access to the same training data (or identical local copies).
3. `torchrun` is executed on each node with consistent environment variables.

### Environment Variables

When running multi-node, `torchrun` (or your job scheduler) needs to know:
- `MASTER_ADDR`: The IP address of the primary node (rank 0).
- `MASTER_PORT`: A free port on the primary node.
- `NNODES`: Total number of nodes.
- `NODE_RANK`: The index of the current node (0 to NNODES-1).

### Manual Launch Example

On **Node 0** (Primary):
```bash
export MASTER_ADDR="192.168.1.10"
export MASTER_PORT=12345
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m quickmt_train.train configs/your_config.yaml
```

On **Node 1**:
```bash
export MASTER_ADDR="192.168.1.10"
export MASTER_PORT=12345
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m quickmt_train.train configs/your_config.yaml
```

### SLURM Integration

On many clusters, you can use SLURM to manage multi-node jobs. Here is a sample submission script:

```bash
#!/bin/bash
#SBATCH --job-name=quickmt-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu

# Get the address of the master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m quickmt_train.train configs/your_config.yaml
```

## Performance & Efficiency

### 1. Scaling the Batch Size

In **Distributed Data Parallel (DDP)** mode (multi-node or `torchrun`), each GPU process maintains its own independent data loader. This means:

- **Effective Batch Size** = `world_size` × `accum_steps` × `max_tokens_per_batch`.
- When you add more GPUs, your total batch size **increases linearly**.
- **Important**: To maintain identical training dynamics when scaling up, you should either decrease `accum_steps` or increase your learning rate (see the *Linear Scaling Rule*).

> [!NOTE]
> This behavior is different from the single-process `DataParallel` fallback. In `DataParallel`, a single batch is split across GPUs, so the total batch size remains constant.

### 2. Linear Scaling Rule

If you increase your global batch size by a factor of $K$, a common practice is to also increase your learning rate by a factor of $K$ to compensate for the reduced number of weight updates and the more stable gradient estimates.

Conversely, if you want to keep your results identical to a single-GPU run, you should decrease `accum_steps` as you increase `world_size` to keep the **Effective Batch Size** constant.


### 3. Gradient Accumulation
`quickmt-train` handles gradient accumulation correctly in DDP by:
- Disabling gradient synchronization (`require_backward_grad_sync = False`) during accumulation steps to avoid unnecessary network overhead.
- Synchronizing the total token count across all ranks before the optimization step to ensure mathematically correct gradient averaging.

### 4. Mixed Precision & TF32
Enable these in your config for a speed boost on modern NVIDIA GPUs (Ampere and later):
```yaml
precision: "bf16"  # or "fp16"
tf32: true
```
- `bf16` is generally more stable than `fp16` and recommended if your hardware supports it.

### 5. Torch Compile
`quickmt-train` is optimized for `torch.compile`. In DDP mode:
- Compilation happens *before* DDP wrapping to avoid deadlocks.
- `dynamic=True` is used to handle variable sequence lengths.
- Compilation can take several minutes at the start of training; this is normal.

## Error Handling & Robustness

### NCCL Timeouts
By default, `quickmt-train` sets a 10-minute NCCL timeout. This ensures that if a node fails or a deadlock occurs (e.g., due to unbalanced data shards), the job will crash with a clear error rather than hanging indefinitely.

### Data Sharding & Workers
Data is sharded using `itertools.islice` based on `global_worker_id = rank * num_workers + worker_id`. 
- **Warning**: If a corpus is smaller than the total number of shards (total GPUs × total workers), some workers will receive zero data. Ensure your training corpora are sufficiently large or sharded appropriately.

### Checkpointing
Only the **Main Process (Rank 0)** saves checkpoints. Other ranks participate in validation but do not write files to disk.

### Synchronization Points
Implicit barriers are placed at:
- **Tokenizer Training**: Only Rank 0 trains tokenizers; other ranks wait until the files exist.
- **Validation**: All ranks must participate in validation to aggregate metrics correctly.
- **Early Stopping**: The decision to stop is synchronized across all ranks.
