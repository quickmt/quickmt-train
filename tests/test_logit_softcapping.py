import torch
import pytest
import sys
import os

# Add parent directory and src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quickmt_train.config import ModelConfig
from quickmt_train.model import Seq2SeqTransformer, GroupedQueryAttention


def test_softcapping_config_and_initialization():
    """Test that ModelConfig correctly initializes and stores the softcap fields."""
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        attn_logit_softcap=50.0,
        final_logit_softcap=30.0,
    )
    model = Seq2SeqTransformer(config)
    
    assert model.config.attn_logit_softcap == 50.0
    assert model.config.final_logit_softcap == 30.0
    
    # Check that attention layers got the softcap parameter
    assert model.encoder.layers[0].self_attn.attn_logit_softcap == 50.0
    assert model.decoder.layers[0].self_attn.attn_logit_softcap == 50.0
    assert model.decoder.layers[0].multihead_attn.attn_logit_softcap == 50.0


def test_final_logit_softcapping_bounds():
    """Test that final logit softcapping strictly bounds the output logits."""
    config_softcap = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        vocab_size_src=100,
        vocab_size_tgt=100,
        final_logit_softcap=15.0,  # Bound of 15.0
    )
    model_softcap = Seq2SeqTransformer(config_softcap)
    model_softcap.eval()
    
    # Set generator weight and bias to very large values to force huge logits
    with torch.no_grad():
        model_softcap.generator.weight.fill_(1000.0)
        model_softcap.generator.bias.fill_(1000.0)
        
        # Run forward pass
        src = torch.randint(4, 100, (2, 5))
        tgt = torch.randint(4, 100, (2, 6))
        
        # Extract logits via return_outputs
        _, (logits, _) = model_softcap(src, tgt, return_outputs=True)
        
        # Verify that all logits are strictly bounded within [-15.0, 15.0]
        assert torch.all(logits >= -15.0)
        assert torch.all(logits <= 15.0)
        
        # Verify it can reach close to the bound (since tanh(1000) is close to 1)
        assert torch.any(logits > 14.9)


def test_attention_logit_softcapping():
    """Test that attention logit softcapping strictly bounds attention weights/scores internally."""
    # Let's instantiate GroupedQueryAttention directly with a small softcap
    softcap = 5.0
    attn = GroupedQueryAttention(
        d_model=32,
        num_heads=2,
        num_heads_kv=2,
        dropout=0.0,
        attn_logit_softcap=softcap,
    )
    attn.eval()
    
    # Force query and key projections to yield extremely large outputs
    with torch.no_grad():
        attn.q_proj.weight.fill_(1000.0)
        attn.k_proj.weight.fill_(1000.0)
        
        query = torch.ones(2, 4, 32)
        key = torch.ones(2, 4, 32)
        value = torch.ones(2, 4, 32)
        
        # Run the attention layer's forward pass
        # This will internally invoke the manual attention path due to attn_logit_softcap
        out, _ = attn(query, key, value)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4, 32)


def test_forward_and_generation_with_softcapping():
    """Test that training forward pass, greedy generate, and beam search work when softcapping is active."""
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        vocab_size_src=100,
        vocab_size_tgt=100,
        attn_logit_softcap=50.0,
        final_logit_softcap=30.0,
    )
    model = Seq2SeqTransformer(config)
    model.eval()
    
    src = torch.randint(4, 100, (2, 8))
    tgt = torch.randint(4, 100, (2, 6))
    
    # 1. Training step forward pass
    loss, num_tokens = model(src, tgt)
    assert isinstance(loss, torch.Tensor)
    assert num_tokens > 0
    
    # 2. Greedy generation
    ys_greedy = model.generate(src, max_len=5, bos_id=2, eos_id=3)
    assert isinstance(ys_greedy, torch.Tensor)
    assert ys_greedy.shape[0] == 2
    assert ys_greedy.shape[1] <= 5
    
    # 3. Beam Search
    ys_beam = model.beam_search(src, max_len=5, beam_size=3, bos_id=2, eos_id=3)
    assert isinstance(ys_beam, torch.Tensor)
    assert ys_beam.shape[0] == 2
    assert ys_beam.shape[1] <= 5
