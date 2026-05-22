import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory and src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quickmt_train.config import ModelConfig, TrainConfig
from quickmt_train.model import Seq2SeqTransformer


def test_set_dropout_propagation():
    """Verify that calling set_dropout dynamically updates all dropout modules and properties."""
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        dropout=0.1,  # Initial dropout
    )
    model = Seq2SeqTransformer(config)
    model.eval()

    # Verify initial state
    assert model.config.dropout == 0.1
    
    # Check that some nn.Dropout has p=0.1
    found_nn_dropout = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0.1
            found_nn_dropout = True
    assert found_nn_dropout, "Should have found nn.Dropout modules"

    # Check GQA layers
    assert model.encoder.layers[0].self_attn.dropout_p == 0.1
    assert model.decoder.layers[0].self_attn.dropout_p == 0.1
    assert model.decoder.layers[0].multihead_attn.dropout_p == 0.1

    # 1. Update dropout to 0.0
    model.set_dropout(0.0)
    assert model.config.dropout == 0.0
    
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0.0
            
    assert model.encoder.layers[0].self_attn.dropout_p == 0.0
    assert model.decoder.layers[0].self_attn.dropout_p == 0.0
    assert model.decoder.layers[0].multihead_attn.dropout_p == 0.0

    # 2. Update dropout to 0.05
    model.set_dropout(0.05)
    assert model.config.dropout == 0.05
    
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0.05
            
    assert model.encoder.layers[0].self_attn.dropout_p == 0.05
    assert model.decoder.layers[0].self_attn.dropout_p == 0.05
    assert model.decoder.layers[0].multihead_attn.dropout_p == 0.05


def test_train_config_default_decay_dropout():
    """Verify that decay_dropout defaults to True in TrainConfig."""
    cfg = TrainConfig()
    assert cfg.decay_dropout is True
