import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.

    Args:
        d_model (int): The dimension of the model.
        dropout (float): Dropout probability. Defaults to 0.1.
        max_len (int): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added and dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache here to make it torch.compile-friendly
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but identical to HF Llama
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies RoPE to Q and K tensors.
    """
    cos_q = cos[:, :, :q.size(2), :]
    sin_q = sin[:, :, :q.size(2), :]
    cos_k = cos[:, :, :k.size(2), :]
    sin_k = sin[:, :, :k.size(2), :]

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors of d_model dimension.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): The dimension of the model.
        padding_idx (int, optional): Index of the padding token. Defaults to None.
    """

    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, tokens):
        """
        Args:
            tokens (torch.Tensor): Token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens scaled by sqrt(d_model).
        """
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class FeedForward(nn.Module):
    """
    Standard or Gated Feed-Forward network.

    Args:
        d_model (int): The dimension of the model.
        ffn_dim (int): The dimension of the feed-forward network's hidden layer.
        dropout (float): Dropout probability. Defaults to 0.1.
        activation (str): Activation function to use ('gelu', 'silu', 'swish', or 'relu'). Defaults to "gelu".
        bias (bool): Whether to use bias in linear layers. Defaults to False.
        mlp_type (str): Type of MLP ('standard' or 'gated'). Defaults to "standard".
    """

    def __init__(
        self,
        d_model,
        ffn_dim,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
    ):
        super().__init__()
        self.mlp_type = mlp_type

        if mlp_type == "gated":
            actual_ffn_dim = int(2 / 3 * ffn_dim)
            self.gate_up_proj = nn.Linear(d_model, 2 * actual_ffn_dim, bias=bias)
            self.down_proj = nn.Linear(actual_ffn_dim, d_model, bias=bias)
        else:
            self.linear1 = nn.Linear(d_model, ffn_dim, bias=bias)
            self.linear2 = nn.Linear(ffn_dim, d_model, bias=bias)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for FeedForward.
        """
        if self.mlp_type == "gated":
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            x = self.act(gate) * up
            x = self.dropout(x)
            x = self.down_proj(x)
        else:
            x = self.act(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
        return x


def get_norm(d_model, eps, bias, norm_type):
    """
    Helper function to get normalization layer.
    """
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model, eps=eps)
    else:
        return nn.LayerNorm(d_model, eps=eps, bias=bias)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention module matching the interface of nn.MultiheadAttention.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        num_heads_kv,
        dropout=0.1,
        bias=False,
        batch_first=True,
        use_rope=False,
        attn_logit_softcap=None,
    ):
        super().__init__()
        assert batch_first, "Only batch_first=True is supported"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.d_model = d_model
        self.batch_first = batch_first
        self.use_rope = use_rope
        self.attn_logit_softcap = attn_logit_softcap

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_heads_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_heads_kv * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
        self.dropout_p = dropout

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
    ):
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (B, num_heads, L, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_heads_kv, self.head_dim).transpose(1, 2)

        if self.use_rope:
            seq_len = max(q_len, kv_len)
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        num_kv_groups = self.num_heads // self.num_heads_kv
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        mask = None
        if attn_mask is not None or key_padding_mask is not None:
            if attn_mask is not None:
                # PyTorch MHA bool mask: True = ignore
                if attn_mask.dtype == torch.bool:
                    m = attn_mask.to(torch.bool).logical_not().unsqueeze(0).unsqueeze(0)
                else:
                    m = attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                m = None

            if key_padding_mask is not None:
                # True = ignore
                km = (
                    key_padding_mask.to(torch.bool)
                    .logical_not()
                    .unsqueeze(1)
                    .unsqueeze(2)
                )
                if m is not None:
                    # combine bool masks
                    if m.dtype == torch.bool and km.dtype == torch.bool:
                        m = m & km
                    else:
                        m = m.to(torch.bool) & km
                else:
                    m = km
            mask = m
            is_causal = False

        if self.attn_logit_softcap is not None:
            # Manual attention with softcapping
            # Compute raw attention logits
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply logit softcapping
            inv_attn_softcap = 1.0 / self.attn_logit_softcap
            attn_scores = self.attn_logit_softcap * torch.tanh(attn_scores * inv_attn_softcap)
            
            # Apply causal mask if specified
            if is_causal:
                causal_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril()
                fill_value = -1e4 if attn_scores.dtype == torch.float16 or attn_scores.dtype == torch.bfloat16 else -1e9
                attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), fill_value)
            
            # Apply padding/attention mask if specified
            if mask is not None:
                if mask.dtype == torch.bool:
                    fill_value = -1e4 if attn_scores.dtype == torch.float16 or attn_scores.dtype == torch.bfloat16 else -1e9
                    attn_scores = attn_scores.masked_fill(~mask, fill_value)
                else:
                    attn_scores = attn_scores + mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            
            # Dropout
            if self.training and self.dropout_p > 0.0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
                
            # Context vector
            out = torch.matmul(attn_weights, v)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.num_heads * self.head_dim)
        )
        out = self.out_proj(out)

        return out, None


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with pre-normalization.
    """

    def __init__(
        self,
        d_model,
        nhead,
        ffn_dim,
        layernorm_eps,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        norm_type="layernorm",
        n_kv_heads=None,
        use_rope=False,
        attn_logit_softcap=None,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_model,
            nhead,
            n_kv_heads or nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            use_rope=use_rope,
            attn_logit_softcap=attn_logit_softcap,
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm2 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Forward pass for EncoderLayer.
        """
        # Pre-norm
        x = self.norm1(src)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        src = src + self.dropout(x)

        x = self.norm2(src)
        x = self.ffn(x)
        src = src + self.dropout(x)
        return src


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with pre-normalization and cross-attention.
    """

    def __init__(
        self,
        d_model,
        nhead,
        ffn_dim,
        layernorm_eps,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        norm_type="layernorm",
        n_kv_heads=None,
        use_rope=False,
        attn_logit_softcap=None,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_model,
            nhead,
            n_kv_heads or nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            use_rope=use_rope,
            attn_logit_softcap=attn_logit_softcap,
        )
        self.multihead_attn = GroupedQueryAttention(
            d_model,
            nhead,
            n_kv_heads or nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            use_rope=False,
            attn_logit_softcap=attn_logit_softcap,
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm2 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm3 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        """
        Forward pass for DecoderLayer.
        """
        # Pre-norm
        x = self.norm1(tgt)
        # Self attention
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=tgt_is_causal,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        # Cross attention
        x = self.norm2(tgt)
        x = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_causal=memory_is_causal,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        # FFN
        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = tgt + self.dropout(x)
        return tgt
