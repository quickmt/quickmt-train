# Getting Started Training a NMT Model With `quickmt-training`

## Overview

Suppose you want to train a NMT model for Icelandic->English translation. 

Before starting, ensure you have:

- **Python 3.12** (recommended)
- **CUDA-capable GPU** (training on CPU is impractical)
- **Git** for cloning the repository
- **Conda** (optional but recommended for environment management)

## 0 - Install `quickmt-train`

First make sure you have `quickmt-train` installed:

```bash
# We suggest setting up a fresh conda environment
conda create -n quickmt-train python=3.12 pip
conda activate quickmt-train
git clone https://github.com/quickmt/quickmt-train.git

# Basic install
pip install -e ./quickmt-train/

# To install aim (for experiment tracking)
pip install -e ./quickmt-train[track]

# To install optuna (for hyperparameter optimization)
pip install -e ./quickmt-train[optimize]

# It may be necessary to downgrade torch to 2.7.1 for torch.compile to work
pip install torch==2.7.1
```

## 1 - Downloading Data

Next, download some pre-processed and filtered data from Huggingface.

```bash
# For faster download speed, log in
hf auth login

# Download training and validation data
quickmt-dataset-to-files quickmt/quickmt-train.is-en is en
quickmt-dataset-to-files quickmt/quickmt-valid.is-en is en
```

This will download four files into your current directory:

* quickmt-train.is-en.is
* quickmt-train.is-en.en
* quickmt-valid.is-en.is
* quickmt-valid.is-en.en


## 2 - Create Config File

Now we will need to create a configuration file. Copy the following into a file named "isen-tiny-1.yaml":

```yaml
train:
  experiment_name: "isen-tiny-1"
  lr: 1.0e-3
  accum_steps: 6
  warmup_steps: 5000
  max_steps: 20000
  eval_steps: 1000
  max_checkpoints: 10
  precision: "bfloat16" # or float16 with an older GPU
  enable_torch_compile: true 

data:
  src_lang: "is"
  tgt_lang: "en"
  src_dev_path: "quickmt-valid.is-en.is"
  tgt_dev_path: "quickmt-valid.is-en.en"
  max_tokens_per_batch: 20000 # tune this value for your GPU memory
  corpora: 
    - src_file: "quickmt-train.is-en.is"
      tgt_file: "quickmt-train.is-en.en"

model:
  d_model: 384
  enc_layers: 6
  dec_layers: 2
  n_heads: 4
  ffn_dim: 1536
  max_len: 256
  vocab_size_src: 12000
  vocab_size_tgt: 12000
```

For all available configuration options, see [`config.py`](https://github.com/quickmt/quickmt-train/blob/main/src/quickmt_train/config.py) or the [Configuration Reference](https://quickmt.github.io/quickmt-train/reference/config).

The `model` section defines the Transformer architecture:

- `d_model`: Model dimension (embedding size)
- `enc_layers` / `dec_layers`: Number of encoder/decoder layers
- `n_heads`: Number of attention heads
- `ffn_dim`: Feed-forward network dimension
- `max_len`: Maximum sequence length
- `vocab_size_src` / `vocab_size_tgt`: Vocabulary sizes for source and target languages

The "tiny" configuration above creates a lightweight model suitable for experimentation. For production use, consider larger values (e.g., `d_model: 768`, `enc_layers: 12`).

If you get an out of memory error, decrease `max_tokens_per_batch` until your batches fit within your GPU memory limits.

### Understanding Batch Size

The **effective batch size** in tokens is calculated as:

```
effective_batch_size = accum_steps × max_tokens_per_batch
```

- **`accum_steps`**: Number of batches to accumulate before updating weights
- **`max_tokens_per_batch`**: Maximum tokens per forward pass (tune for your GPU memory)

For this config: `6 × 20000 = 120,000` tokens per optimization step.

Gradient accumulation allows simulating larger batch sizes without additional GPU memory.

## 3 - Train

Next we train our model!

```bash
quickmt-train --config isen-tiny-1.yaml
```


During training, the following will be created in your experiment directory (`./isen-tiny-1/`):

- `checkpoints/` - Model checkpoints saved every `eval_steps` iterations
- `tokenizer_src.model` / `tokenizer_tgt.model` - Trained SentencePiece tokenizers
- `aim-runs/` - Experiment tracking data (if using aim)

Training progress is logged to the console. You can monitor GPU utilization with `nvidia-smi` in a separate terminal.

### Troubleshooting

Out of Memory (OOM) Errors

- Decrease `max_tokens_per_batch` (try 10000, then 5000)
- Reduce `max_len` in model config
- Disable `torch.compile` temporarily

`torch.compile` Errors

- Try installing torch version 2.7.1
- Disable `torch.compile`

Slow Training

- Ensure `enable_torch_compile: true` (first run is slow due to compilation)
- Increase `max_tokens_per_batch` if memory allows
- Check GPU utilization with `nvidia-smi`
- Buy or rent a faster GPU 


## 4 - Model Export

Once the model is trained, or at any time during training, you can export and evaluate your model:

```bash
# Average checkpoints (and optionally quantize the model)
# You can over-ride values in your config file by passing them as arguments, e.g. --k 1
quickmt-avg --experiment_dir ./isen-tiny-1 --k 1

# Convert to CTranslate2 format
quickmt-export --experiment_dir ./isen-tiny-1
```

## 5 - Model Evaluation

```bash
# Download Flores Data
# NOTE: You will need to log in to Huggingface
# And accept the flores-plus terms of use: https://huggingface.co/datasets/openlanguagedata/flores_plus
quickmt-flores-download isl_Latn eng_Latn

# Evaluate (uses quickmt library, https://github.com/quickmt/quickmt)
quickmt-eval --src_file flores_plus_isl_Latn.txt --ref_file flores_plus_eng_Latn.txt --device cuda --batch_size 32 --beam_size 5 --model ./isen-tiny-1/exported_model
```

## 6 - Done!

Now you have a trained NMT model for Icelandic->English translation. Try tweaking the config file and training again! If you have any trouble raise an issue in our [github repo](https://github.com/quickmt/quickmt-train/issues).
