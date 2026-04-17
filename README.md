# NMT Model Training From Scratch 

Experimenting with training Neural Machine Translation (NMT) models from scratch using PyTorch.


## WARNING - WORK IN PROGRESS!!!

This project is an active work in progress. There is still some work to be done:

* Validation metrics do not seem to be calculated correctly
* The `generate` and `beam_search` methods in `model.py` does not seem to be implemented correctly
* ... etc (see open issues)


## Key Features


### 🚀 Performance & Optimization

- **Flash Attention**: Automatic Flash Attention support via `scaled_dot_product_attention` (2-4x faster attention)
- **`torch.compile`**: Integrated torch compilation for faster training
- **Mixed Precision (AMP)**: Uses `torch.autocast` with `bfloat16` or `float16` for faster training and reduced memory usage
- **Gradient Checkpointing**: Trade compute for memory to enable larger batch sizes (40-60% memory reduction)
- **Gradient Accumulation & Clipping**: Support for large effective batch sizes and stable training via gradient norm scaling
- **Efficient Attention**: Optimized attention mechanisms with reduced overhead
- **DDP Optimizations**: Distributed training with optimized communication patterns
- **Tensor Core Optimization**: Sequence padding to multiples of 8/16 for maximum GPU utilization

### 📊 Data Processing

- **Streaming Dataset**: `IterableDataset` implementation for handling datasets larger than RAM
- **Token-Based Batching**: Dynamic batching with bucket sorting to minimize padding and maximize throughput
- **SentencePiece Tokenization**: Integrated support for training and on-the-fly SentencePiece (unigram/BPE) tokenization
- **Multi-worker Sharding**: Efficient data loading with automatic sharding across multiple CPU workers
- **Optimized Buffering**: Improved buffering strategy for reduced CPU bottlenecks
* **Multi-dataset training**: Train on multiple datasets at once starting/stopping at specific steps

### 📈 Evaluation & Monitoring

- **Real-time Logging**: Tracking of Loss, Perplexity (PPL), Token Accuracy etc.
- **Translation Quality**: In-training evaluation using **BLEU** and **ChrF** scores via `sacrebleu`
- **Aim Tracking**: Integration with `aim` for experiment tracking and visualization
- **Hyperparameter Optimization**: Integration with `optuna` for hyperparameter optimization

### 🛠️ Inference & Deployment

- **Model Averaging**: Tool for stochastic weight averaging of multiple checkpoints to improve generalization
- **CTranslate2 Export**: Script to convert PyTorch models to CTranslate2 format for production deployment
- **quickmt compatible**: Models can be used with the quickmt library for inference

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/quickmt/quickmt-train.git
cd quickmt-train

# Install with uv (creates virtual environment and installs dependencies)
uv sync

# Or install with all optional extras (docs, tracking, optimization, dev tools)
uv sync --all-extras
```

### Using conda/mamba

```bash
conda create -n quickmt-train python=3.12 pip
conda activate quickmt-train
git clone https://github.com/quickmt/quickmt-train.git
pip install -e ./quickmt-train/
```


## Usage

See the [quickstart](https://quickmt.github.io/quickmt-train/quickstart/) for a complete example.

```bash
# Create a config file (see the examples)
vim configs/faen-tiny.yaml

# Train
quickmt-train --config configs/faen-tiny.yaml 

# Average checkpoints and quantize the model
quickmt-avg --experiment_dir ./faen-tiny   

# Convert to CTranslate2 format
quickmt-export --experiment_dir ./faen-tiny   

# Evaluate (uses quickmt library, https://github.com/quickmt/quickmt)
quickmt-eval --src_file data/flores.fa --ref_file data/flores.en --device cuda --batch_size 16 --beam_size 5 --model ./faen-tiny/exported_model
```
