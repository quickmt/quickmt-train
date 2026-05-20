import torch
import torch.nn as nn
import pytest
import os
import sys
import tempfile
import shutil
from safetensors.torch import save_file

# Add parent directory and src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quickmt_train.config import ModelConfig, DataConfig, TrainConfig, ExportConfig
from quickmt_train.model import Seq2SeqTransformer, RotaryEmbedding
from quickmt_train.convert_to_ct2 import convert_to_ct2_cli


def test_rope_initialization():
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        use_rope=True,
    )
    model = Seq2SeqTransformer(config)
    
    # Verify that self positional encoding is nn.Dropout and not PositionalEncoding
    assert isinstance(model.positional_encoding, nn.Dropout)
    
    # Verify use_rope settings in GQA
    enc_self_attn = model.encoder.layers[0].self_attn
    assert enc_self_attn.use_rope is True
    assert isinstance(enc_self_attn.rotary_emb, RotaryEmbedding)
    
    dec_self_attn = model.decoder.layers[0].self_attn
    assert dec_self_attn.use_rope is True
    assert isinstance(dec_self_attn.rotary_emb, RotaryEmbedding)
    
    dec_cross_attn = model.decoder.layers[0].multihead_attn
    assert dec_cross_attn.use_rope is False
    assert not hasattr(dec_cross_attn, "rotary_emb")


def test_rope_forward_and_generate():
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        use_rope=True,
        vocab_size_src=100,
        vocab_size_tgt=100,
    )
    model = Seq2SeqTransformer(config)
    model.eval()
    
    # Dummy input
    src = torch.randint(4, 100, (2, 10))
    tgt = torch.randint(4, 100, (2, 8))
    
    # Forward pass (training step)
    loss, num_tokens = model(src, tgt)
    assert isinstance(loss, torch.Tensor)
    assert num_tokens > 0
    
    # Generate pass
    ys = model.generate(src, max_len=5, bos_id=2, eos_id=3)
    assert isinstance(ys, torch.Tensor)
    assert ys.shape[0] == 2
    assert ys.shape[1] <= 5


def test_rope_ct2_conversion():
    config = ModelConfig(
        d_model=64,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        ffn_dim=128,
        max_len=32,
        use_rope=True,
        vocab_size_src=100,
        vocab_size_tgt=100,
    )
    model = Seq2SeqTransformer(config)
    model.eval()
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configs
        import yaml
        config_path = os.path.join(temp_dir, "config.yaml")
        
        cfg_dict = {
            "model": {
                "d_model": 64,
                "enc_layers": 1,
                "dec_layers": 1,
                "n_heads": 2,
                "ffn_dim": 128,
                "max_len": 32,
                "use_rope": True,
                "vocab_size_src": 100,
                "vocab_size_tgt": 100,
                "tie_decoder_embeddings": False,
                "joint_vocab": False,
            },
            "data": {
                "experiment_name": temp_dir,
                "src_lang": "fa",
                "tgt_lang": "en",
            },
            "train": {
                "experiment_name": temp_dir,
            },
            "export": {
                "quantization": "none",
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)
            
        # Create fake tokenizer vocab files
        for suffix in ["_src", "_tgt"]:
            vocab_path = os.path.join(temp_dir, f"tokenizer{suffix}.vocab")
            with open(vocab_path, "w", encoding="utf-8") as f:
                # Add 100 fake tokens
                for idx in range(100):
                    f.write(f"token_{idx}\t-{idx}\n")
            model_path = os.path.join(temp_dir, f"tokenizer{suffix}.model")
            with open(model_path, "w") as f:
                f.write("fake model content")
                
        # Save model safetensors
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, "model_001.safetensors")
        save_file(model.state_dict(), checkpoint_path)
        
        # tokenizer_prefix_src/tgt are properties derived from experiment_name, so we don't pass them in yaml
            
        # Convert using our convert_to_ct2_cli
        # We let the exporter use the default output_dir, which is under temp_dir (exported_model)
        out_dir = os.path.join(temp_dir, "exported_model")
        
        import ctranslate2
        # Let's run convert_to_ct2_cli
        convert_to_ct2_cli(temp_dir)
        
        # Verify that output files exist
        assert os.path.exists(os.path.join(out_dir, "model.bin"))
        assert os.path.exists(os.path.join(out_dir, "shared_vocabulary.json"))
        
        # Load in ctranslate2 and translate
        translator = ctranslate2.Translator(out_dir, device="cpu")
        assert translator is not None
