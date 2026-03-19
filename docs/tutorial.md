# Getting Started Training a NMT Model With `quickmt-training`

## Overview

Suppose you want to train a NMT model for Ukranian->English translation. The first step will be identifying which data to use to train your model. 

## 1 - Downloading Data

[`mt-data`](https://github.com/thammegowda/mtdata) is an excellent tool for managing and curating parallel corpora for NMT training, and we'll show how to use it to download NMT training data. First use the `mtdata list` to show the available datasets:

```bash
mtdata list --langs ukr-eng
```