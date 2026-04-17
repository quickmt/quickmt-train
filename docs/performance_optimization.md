# Performance Optimization Guide

This document describes the performance optimizations implemented in quickmt-train for faster training.

## Overview

The following optimizations have been implemented to maximize training throughput and efficiency:

1. **Flash Attention Support**
2. **Efficient Scaled Dot-Product Attention**
3. **Gradient Checkpointing**
4. **Optimized Data Loading Pipeline**
5. **Improved Gradient Accumulation**
6. **DDP Communication Optimizations**
7. **Validation Loop Optimizations**
8. **Memory Efficiency Improvements**

---

## 1. Flash Attention Support

### What it does
Uses PyTorch's `scaled_dot_product_attention` which automatically leverages Flash Attention on supported GPUs (Ampere architecture and newer).

### Benefits
- **2-4x faster attention computation**
- **Reduced memory usage** (doesn't materialize attention matrix)
- **Automatic fallback** to standard attention on unsupported hardware

### Implementation
- `model.py`: EncoderLayer and DecoderLayer use `_efficient_attention` methods
- Automatically detects and uses Flash Attention when available

### Configuration
```yaml
model:
  use_flash_attention: true
  use_efficient_attention: true
```

### Requirements
- PyTorch 2.0+
- GPU with Compute Capability 8.0+ (Ampere) for Flash Attention
- CUDA 11.6+

---

## 2. Efficient Scaled Dot-Product Attention

### What it does
Replaces `nn.MultiheadAttention` with direct `F.scaled_dot_product_attention` calls, avoiding overhead from the wrapper class.

### Benefits
- **10-15% faster attention** by avoiding function call overhead
- **Better kernel fusion** opportunities
- **Reduced memory allocations**

### Implementation
- Custom `_efficient_attention_self` and `_efficient_attention_cross` methods in EncoderLayer/DecoderLayer
- Direct tensor operations without wrapper overhead

---

## 3. Gradient Checkpointing

### What it does
Activates recomputation during backward pass to trade compute for memory. Instead of storing all activations, recomputes them during backpropagation.

### Benefits
- **40-60% memory reduction** for encoder/decoder activations
- **Enables larger batch sizes** or deeper models
- **~20% compute overhead** (acceptable trade-off for memory-bound training)

### Implementation
- `model.py`: Sets `gradient_checkpointing = True` on encoder/decoder when configured
- Native PyTorch gradient checkpointing via `torch.utils.checkpoint`

### Configuration
```yaml
model:
  use_checkpoint: true
```

### When to use
- Training large models (512+ d_model)
- Memory-limited scenarios
- When batch size is constrained by memory rather than compute

### When to disable
- Small models where memory isn't a bottleneck
- When training is compute-bound (not memory-bound)

---

## 4. Optimized Data Loading Pipeline

### What it does
Streamlines data loading and tokenization to minimize CPU bottlenecks during training.

### Benefits
- **Faster tokenization** with dedicated helper function
- **Better buffering strategy** reduces stalls
- **Improved sharding** for multi-GPU training

### Key Optimizations

#### 4.1 Tokenization Helper
```python
def tokenize_pair(s, t):
    """Efficiently tokenize a pair of sentences."""
    # Pre-defined function avoids repeated lookups
    s_ids = self.src_sp.encode(...)
    t_ids = self.tgt_sp.encode(...)
    return s_ids, t_ids
```

#### 4.2 Efficient Buffering
- Larger default buffer size (20,000 minimum)
- Better shuffling strategy to reduce length bias
- Sort by sequence length to minimize padding

#### 4.3 Multi-Worker Loading
```yaml
data:
  num_workers: 4
  prefetch_factor: 64
  persistent_workers: true
```

### Recommendations
- Set `num_workers` to 2-4× number of GPUs
- Increase `prefetch_factor` for slow storage (64-128)
- Use SSDs for dataset storage

---

## 5. Improved Gradient Accumulation

### What it does
Optimizes the gradient accumulation process to reduce communication overhead and improve throughput.

### Benefits
- **Reduced synchronization points** in DDP
- **Better token-based scaling** for variable-length sequences
- **More efficient optimizer steps**

### Implementation
```python
# Use no_sync context to avoid unnecessary DDP sync during accumulation
if world_size > 1 and (batch_idx + 1) % train_cfg.accum_steps != 0:
    context = model.no_sync()
else:
    context = nullcontext()

with context:
    scaler.scale(loss).backward()
```

### Configuration
```yaml
train:
  accum_steps: 30  # Adjust based on GPU memory
  grad_clip: 1.0
```

### Best Practices
- Set `accum_steps` to maximize GPU utilization
- Monitor GPU memory - increase if underutilized
- Typical values: 16-64 depending on model size

---

## 6. DDP Communication Optimizations

### What it does
Reduces overhead in DistributedDataParallel training by optimizing synchronization patterns.

### Benefits
- **Faster all-reduce operations** for gradient synchronization
- **Reduced communication overhead** in multi-GPU setups
- **Better scaling** with increasing GPU count

### Key Optimizations

#### 6.1 DDP Configuration
```python
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False,  # Skip unused parameter sync
    gradient_as_bucket_view=True,   # Reduce memory copies
    broadcast_buffers=False,        # Avoid buffer broadcasts
)
```

#### 6.2 Token Count Synchronization
```python
# Efficient token count aggregation across GPUs
if world_size > 1:
    token_tensor = torch.tensor([accum_tokens], device=device, dtype=torch.float32)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    global_accum_tokens = token_tensor.item()
```

#### 6.3 Gradient Scaling
```python
# Scale gradients by world_size before token-based normalization
p.grad.data.mul_(world_size).div_(max(1.0, global_accum_tokens))
```

### Multi-GPU Training Tips
- Use `torchrun` for launching distributed training
- Ensure NCCL backend is available (`torch.distributed` uses it by default)
- Monitor GPU utilization with `nvidia-smi dmon`

---

## 7. Validation Loop Optimizations

### What it does
Streamlines the validation process to reduce overhead and improve throughput.

### Benefits
- **Faster validation** with teacher-forced predictions
- **Batch processing** for generation samples
- **Reduced memory usage** during metric computation

### Key Optimizations

#### 7.1 Teacher-Forced Predictions
```python
# Use logits from forward pass instead of autoregressive generation
generated_ids = preds  # Much faster than model.generate()
```

#### 7.2 Efficient Sample Processing
```python
# Process batch at once instead of one-by-one
batch_size = src.size(0)
for i in range(min(batch_size, max_samples - sample_count)):
    # Process sample
```

#### 7.3 Inference Mode
```python
with torch.inference_mode():  # Better than torch.no_grad()
    # Validation code
```

### Configuration
```yaml
train:
  val_max_samples: 500  # Limit BLEU computation
  quick_test_samples: 5  # Final test samples
  eval_steps: 1000  # Validation frequency
```

---

## 8. Memory Efficiency Improvements

### What it does
Various optimizations to reduce memory usage and enable larger batch sizes.

### 8.1 Optimizer State Management
```python
# More efficient gradient clearing
optimizer.zero_grad(set_to_none=True)  # Saves memory vs zero_grad()
```

### 8.2 Mixed Precision Training
```yaml
train:
  precision: "bf16"  # or "fp16"
  tf32: true  # Enable TF32 on Ampere+
```

**BF16 vs FP16:**
- **BF16**: Larger dynamic range, more stable, recommended
- **FP16**: Requires loss scaling, may overflow on large models

### 8.3 Tensor Core Optimization
```python
# Pad sequences to multiples of 8/16 for Tensor Cores
def pad_to_multiple(tensor, multiple=16):
    # Padding logic
```

### 8.4 cuDNN Benchmark
```python
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matrix multiplication
```

---

## Performance Tuning Checklist

### Hardware-Specific
- [ ] Enable TF32 for Ampere+ GPUs (`tf32: true`)
- [ ] Use BF16 precision on Ampere+ GPUs
- [ ] Set `pad_multiple: 8` or `16` for Tensor Cores
- [ ] Ensure PCIe/NVLink bandwidth is sufficient for multi-GPU

### Model-Specific
- [ ] Enable gradient checkpointing for large models (`use_checkpoint: true`)
- [ ] Use Flash Attention (`use_flash_attention: true`)
- [ ] Set appropriate `accum_steps` based on GPU memory
- [ ] Consider RMSNorm over LayerNorm for slightly better performance

### Data Pipeline
- [ ] Set `num_workers: 4-8` based on CPU count
- [ ] Increase `prefetch_factor: 64-128` for slow storage
- [ ] Use `buffer_size: 20000+` for better shuffling
- [ ] Store datasets on SSDs, not HDDs

### Distributed Training
- [ ] Use `torchrun --nproc_per_node=N` for multi-GPU
- [ ] Set `find_unused_parameters=False` in DDP
- [ ] Enable `gradient_as_bucket_view=True` in DDP
- [ ] Monitor GPU utilization to identify bottlenecks

### Training Loop
- [ ] Enable `torch.compile` for single-GPU training
- [ ] Use fused AdamW optimizer (`fused=True`)
- [ ] Set appropriate `grad_clip: 1.0` to prevent exploding gradients
- [ ] Log every 100 steps (`log_steps: 100`) to reduce overhead

---

## Benchmarking Performance

Use the built-in benchmark utility to measure performance:

```bash
quickmt-train --config configs/uken-base.yaml --benchmark
```

Or manually measure:
```python
from quickmt_train.benchmark import benchmark
benchmark(model_cfg, data_cfg, train_cfg)
```

**Key Metrics:**
- **tok/s**: Tokens per second (higher is better)
- **steps/sec**: Training steps per second
- **GPU memory**: Peak memory usage (lower allows larger batches)

---

## Troubleshooting

### Issue: Out of Memory
**Solutions:**
1. Enable gradient checkpointing: `use_checkpoint: true`
2. Reduce `max_tokens_per_batch`
3. Increase `accum_steps`
4. Use smaller model dimensions

### Issue: Low GPU Utilization (<50%)
**Solutions:**
1. Increase `num_workers` in data config
2. Increase `prefetch_factor`
3. Check if data loading is the bottleneck (use `nvtop` or `nvidia-smi`)
4. Ensure dataset is on fast storage (SSD)

### Issue: Slow Multi-GPU Scaling
**Solutions:**
1. Check NVLink/PCIe connectivity
2. Reduce communication frequency (increase `accum_steps`)
3. Use `gradient_as_bucket_view=True`
4. Ensure NCCL is using the optimal backend

### Issue: Flash Attention Not Working
**Solutions:**
1. Verify PyTorch version >= 2.0
2. Check GPU compute capability >= 8.0 (Ampere)
3. Ensure CUDA version >= 11.6
4. Falls back automatically to standard attention if unsupported

---

## Expected Performance Improvements

Based on typical training scenarios:

| Optimization | Speed Improvement | Memory Improvement |
|--------------|-------------------|-------------------|
| Flash Attention | 30-50% | 20-40% |
| Gradient Checkpointing | -20% (compute) | 40-60% |
| Efficient Attention | 10-15% | 5-10% |
| Optimized Data Loading | 5-20% | - |
| Mixed Precision (BF16) | 40-80% | 50% |
| DDP Optimizations | 10-30% (multi-GPU) | 10-20% |
| **Combined** | **2-4x** | **2-3x larger batches** |

*Note: Actual improvements depend on hardware, model size, and dataset characteristics.*

---

## Future Optimizations

Potential areas for future improvement:

1. **Tensor Parallelism**: For very large models (>1B parameters)
2. **Pipeline Parallelism**: Split model across GPUs
3. **Activation Compression**: Use FP8 or INT8 activations
4. **Kernel Fusion**: Custom fused operators for specific architectures
5. **Asynchronous Data Loading**: Overlap data loading with forward pass
6. **Dynamic Batch Sizing**: Adjust batch size based on sequence length distribution

---

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Efficient Training Techniques](https://huggingface.co/docs/transformers/performance)
- [DDP Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
