# Getting Started Training a NMT Model With `quickmt-training`

## Overview

Suppose you want to train a NMT model for Icelandic->English translation. 

## 0 - Install `quickmt-train`

First make sure you have `quickmt-train` installed:

```bash
# We suggest setting up a fresh conda environment
conda create -n quickmt-train python=3.12 pip
conda activate quickmt-train
git clone https://github.com/quickmt/quickmt-train.git
pip install -e ./quickmt-train/
```

## 1 - Downloading Data

Next, download some pre-processed and filtered data from Huggingface.

```bash
quickmt-dataset-to-files quickmt/quickmt-train.is-en is en
quickmt-dataset-to-files quickmt/newscrawl2024-en-backtranslated-is is en 
quickmt-dataset-to-files quickmt/quickmt-valid.id-en is en
```

## 2 - Create Config File

Now we will need to create a configuration file. Copy the following into a file named "isen-tiny-1.yaml":

```yaml
train:
  experiment_name: "isen-tiny-1"
  lr: 2.5e-3
  accum_steps: 3
  warmup_steps: 5000
  max_steps: 40000
  eval_steps: 1000
  max_checkpoints: 10
  precision: "bf16"
  grad_clip: 1.0
  enable_torch_compile: true

data:
  src_lang: "is"
  tgt_lang: "en"
  src_dev_path: "quickmt-valid.id-en.is"
  tgt_dev_path: "quickmt-valid.id-en.en"
  max_tokens_per_batch: 40000
  buffer_size: 50000
  num_workers: 4
  src_spm_nbest_size: 20
  src_spm_alpha: 0.5
  tgt_spm_nbest_size: 20
  tgt_spm_alpha: 0.5
  corpora:
    - src_file: "quickmt-train.is-en.is"
      tgt_file: "quickmt-train.is-en.en"
      weight: 2
      start_step: 0
    - src_file: "newscrawl2024-en-backtranslated-is.is"
      tgt_file: "newscrawl2024-en-backtranslated-is.en"
      weight: 1
      start_step: 0

model:
  d_model: 384
  enc_layers: 6
  dec_layers: 2
  n_heads: 4
  ffn_dim: 1536
  max_len: 512
  vocab_size_src: 16000
  vocab_size_tgt: 16000
  mlp_type: "standard"
  activation: "gelu"
  dropout: 0.1
```

If you get an out of memory error, decrease `max_tokens_per_batch` until your batches fit within your GPU memory limits. `quickmt-train`. 

The "effective batch size" in tokens is <Number of GPUs> * <accum_steps> * <max_tokens_per_batch>.


## 3 - Train

Next we train our model!

```bash
quickmt-train --config isen-tiny-1.yaml
```

If you have trouble with `torch.compile` you can disable it (not recommended) by setting `enable_torch_compile: false` in the config.


## 4 - Export and Evaluate

Once the model is trained, or periodically if you are impatient, export and evaluate your model:

```bash
# Average checkpoints and quantize the model
quickmt-avg --experiment_dir ./isen-tiny-1

# Convert to CTranslate2 format
quickmt-export --experiment_dir ./isen-tiny-1

# Download Flores Data
# NOTE: You will need to log in to Huggingface
# And accept the flores-plus terms of use: https://huggingface.co/datasets/openlanguagedata/flores_plus
quickmt-flores-download isl_Latn eng_Latn

# Evaluate (uses quickmt library, https://github.com/quickmt/quickmt)
quickmt-eval --src_file flores_plus_isl_Latn.txt --ref_file flores_plus_eng_Latn.txt --device cuda --batch_size 16 --beam_size 5 --model ./isen-tiny-1/exported_model
```