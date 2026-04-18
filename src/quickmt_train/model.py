import torch
import torch.nn as nn
import math
import torch.ao.quantization


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
            self.gate_up_proj = nn.Linear(d_model, 2 * ffn_dim, bias=bias)
            self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
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

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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

    Args:
        d_model (int): The dimension of the model.
        eps (float): A small value for numerical stability.
        bias (bool): Whether to use bias in the normalization layer.
        norm_type (str): Type of normalization ('rmsnorm' or 'layernorm').

    Returns:
        nn.Module: The normalization layer.
    """
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model, eps=eps)
    else:
        return nn.LayerNorm(d_model, eps=eps, bias=bias)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention module matching the interface of nn.MultiheadAttention.
    """
    def __init__(self, d_model, num_heads, num_heads_kv, dropout=0.1, bias=False, batch_first=True):
        super().__init__()
        assert batch_first, "Only batch_first=True is supported"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.d_model = d_model
        self.batch_first = batch_first
        
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_heads_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_heads_kv * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
        self.dropout_p = dropout

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

        num_kv_groups = self.num_heads // self.num_heads_kv
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        mask = None
        if attn_mask is not None or key_padding_mask is not None:
            if attn_mask is not None:
                # PyTorch MHA bool mask: True = ignore
                if attn_mask.dtype == torch.bool:
                    m = (attn_mask == False).unsqueeze(0).unsqueeze(0)
                else:
                    m = attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                m = None

            if key_padding_mask is not None:
                # True = ignore
                km = (key_padding_mask == False).unsqueeze(1).unsqueeze(2)
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

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        
        return out, None


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with pre-normalization.

    Args:
        d_model (int): The dimension of the model.
        nhead (int): The number of attention heads.
        ffn_dim (int): The dimension of the feed-forward network.
        layernorm_eps (float): A small value for numerical stability in normalization layers.
        dropout (float): Dropout probability. Defaults to 0.1.
        activation (str): Activation function for FFN. Defaults to "gelu".
        bias (bool): Whether to use bias in linear and attention layers. Defaults to False.
        mlp_type (str): Type of MLP in FFN. Defaults to "standard".
        norm_type (str): Type of normalization. Defaults to "layernorm".
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
    ):
        super().__init__()
        if n_kv_heads is not None and n_kv_heads != nhead:
            self.self_attn = GroupedQueryAttention(
                d_model, nhead, n_kv_heads, dropout=dropout, bias=bias, batch_first=True
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True, bias=bias
            )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm2 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Forward pass for EncoderLayer.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor, optional): Attention mask for src.
            src_key_padding_mask (torch.Tensor, optional): Padding mask for src.
            is_causal (bool): Whether the attention is causal. Defaults to False.

        Returns:
            torch.Tensor: Processed source sequence.
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

    Args:
        d_model (int): The dimension of the model.
        nhead (int): The number of attention heads.
        ffn_dim (int): The dimension of the feed-forward network.
        layernorm_eps (float): A small value for numerical stability in normalization layers.
        dropout (float): Dropout probability. Defaults to 0.1.
        activation (str): Activation function for FFN. Defaults to "gelu".
        bias (bool): Whether to use bias in linear and attention layers. Defaults to False.
        mlp_type (str): Type of MLP in FFN. Defaults to "standard".
        norm_type (str): Type of normalization. Defaults to "layernorm".
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
    ):
        super().__init__()
        if n_kv_heads is not None and n_kv_heads != nhead:
            self.self_attn = GroupedQueryAttention(
                d_model, nhead, n_kv_heads, dropout=dropout, bias=bias, batch_first=True
            )
            self.multihead_attn = GroupedQueryAttention(
                d_model, nhead, n_kv_heads, dropout=dropout, bias=bias, batch_first=True
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True, bias=bias
            )
            self.multihead_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True, bias=bias
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

        Args:
            tgt (torch.Tensor): Target sequence.
            memory (torch.Tensor): Output from encoder.
            tgt_mask (torch.Tensor, optional): Self-attention mask for tgt.
            memory_mask (torch.Tensor, optional): Cross-attention mask for memory.
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for tgt.
            memory_key_padding_mask (torch.Tensor, optional): Padding mask for memory.
            tgt_is_causal (bool): Whether self-attention is causal. Defaults to False.
            memory_is_causal (bool): Whether cross-attention is causal. Defaults to False.

        Returns:
            torch.Tensor: Processed target sequence.
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


class Seq2SeqTransformer(nn.Module):
    """
    Sequence-to-sequence Transformer model.

    Args:
        config: Configuration object containing model hyperparameters.
            Required attributes: vocab_size_src, vocab_size_tgt, d_model, dropout, max_len,
            n_heads, ffn_dim, layernorm_eps, activation, ff_bias, mlp_type, norm_type,
            enc_layers, dec_layers, tie_decoder_embeddings, pad_id, bos_id, eos_id.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.src_tok_emb = TokenEmbedding(
            config.vocab_size_src, config.d_model, padding_idx=config.pad_id
        )
        self.tgt_tok_emb = TokenEmbedding(
            config.vocab_size_tgt, config.d_model, padding_idx=config.pad_id
        )
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )

        encoder_layer = EncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            ffn_dim=config.ffn_dim,
            layernorm_eps=config.layernorm_eps,
            dropout=config.dropout,
            activation=config.activation,
            bias=config.ff_bias,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.enc_layers,
            norm=get_norm(
                config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
            ),
        )

        decoder_layer = DecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            ffn_dim=config.ffn_dim,
            layernorm_eps=config.layernorm_eps,
            dropout=config.dropout,
            activation=config.activation,
            bias=config.ff_bias,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.dec_layers,
            norm=get_norm(
                config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
            ),
        )

        # Always use bias for the generator
        self.generator = nn.Linear(config.d_model, config.vocab_size_tgt, bias=True)
        if config.tie_decoder_embeddings:
            self.generator.weight = self.tgt_tok_emb.embedding.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Standard Transformer initialization:
        - Xavier Uniform for linear weights (robust default for NMT)
        - Depth-aware Normal init for embeddings (1/sqrt(d_model))
        - Zero for biases and padding indices
        - Unit weight for LayerNorm/RMSNorm
        """
        if isinstance(module, nn.Linear):
            # If weights are tied, the generator weight is just a pointer to embeddings.
            # We skip re-initializing it here to preserve the embedding stats.
            if not (self.config.tie_decoder_embeddings and module is self.generator):
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Using 1/sqrt(d_model) ensures unit variance after the sqrt(d_model) scaling in forward.
            nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model**-0.5)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # PyTorch's MultiheadAttention uses parameters directly
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            # out_proj is a sub-module handled by the Linear case above
        elif isinstance(module, GroupedQueryAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.q_proj.bias is not None:
                nn.init.zeros_(module.q_proj.bias)
                nn.init.zeros_(module.k_proj.bias)
                nn.init.zeros_(module.v_proj.bias)
                nn.init.zeros_(module.out_proj.bias)

    def encode(self, src=None, src_mask=None, src_probs=None):
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence of shape (batch_size, src_len).
            src_mask (torch.Tensor, optional): Encoder attention mask.
            src_probs (torch.Tensor, optional): Soft source probabilities of shape (batch, len, vocab).

        Returns:
            torch.Tensor: Encoded sequence (memory).
        """
        if src_probs is not None:
            src_emb = torch.matmul(src_probs, self.src_tok_emb.embedding.weight)
            src_emb = src_emb * math.sqrt(self.config.d_model)
            src_padding_mask = (src_probs.argmax(dim=-1) == self.config.pad_id).to(torch.bool)
        else:
            src_emb = self.src_tok_emb(src)
            src_padding_mask = (src == self.config.pad_id).to(torch.bool)
            
        src_emb = self.positional_encoding(src_emb)
        # If src_mask is provided (e.g. for specific attention patterns), ensure it's bool
        if src_mask is not None and src_mask.dtype != torch.bool:
            src_mask = (
                (src_mask < 0)
                if src_mask.is_floating_point()
                else src_mask.to(torch.bool)
            )

        # Standard encoder is non-causal
        memory = self.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        return memory

    def decode(
        self,
        tgt=None,
        memory=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
        tgt_probs=None,
    ):
        """
        Decode the target sequence using the encoded source.

        Args:
            tgt (torch.Tensor): Target sequence of shape (batch, tgt_len).
            memory (torch.Tensor): Output from encoder.
            tgt_mask (torch.Tensor, optional): Self-attention mask for tgt.
            memory_mask (torch.Tensor, optional): Cross-attention mask for memory.
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for tgt.
            memory_key_padding_mask (torch.Tensor, optional): Padding mask for memory.
            tgt_is_causal (bool): Whether self-attention is causal.
            memory_is_causal (bool): Whether cross-attention is causal.
            tgt_probs (torch.Tensor, optional): Soft target probabilities.

        Returns:
            torch.Tensor: Decoded features.
        """
        if tgt_probs is not None:
            tgt_emb = torch.matmul(tgt_probs, self.tgt_tok_emb.embedding.weight)
            tgt_emb = tgt_emb * math.sqrt(self.config.d_model)
        else:
            tgt_emb = self.tgt_tok_emb(tgt)
            
        tgt_emb = self.positional_encoding(tgt_emb)

        # Ensure all masks are boolean
        if tgt_mask is not None and tgt_mask.dtype != torch.bool:
            tgt_mask = (
                (tgt_mask < 0)
                if tgt_mask.is_floating_point()
                else tgt_mask.to(torch.bool)
            )
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            memory_mask = (
                (memory_mask < 0)
                if memory_mask.is_floating_point()
                else memory_mask.to(torch.bool)
            )
        if (
            tgt_key_padding_mask is not None
            and tgt_key_padding_mask.dtype != torch.bool
        ):
            tgt_key_padding_mask = tgt_key_padding_mask.to(torch.bool)
        if (
            memory_key_padding_mask is not None
            and memory_key_padding_mask.dtype != torch.bool
        ):
            memory_key_padding_mask = memory_key_padding_mask.to(torch.bool)

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return out

    def project(self, x):
        """
        Project hidden features to vocabulary logits.

        Args:
            x (torch.Tensor): Output from decoder.

        Returns:
            torch.Tensor: Logits of shape (..., vocab_size_tgt).
        """
        return self.generator(x)

    def forward(self, src=None, tgt=None, return_outputs=False, label_smoothing=0.0, src_probs=None):
        """
        Single training step: encode, decode, and calculate loss.

        Args:
            src (torch.Tensor): Source sequences.
            tgt (torch.Tensor): Target sequences (including BOS and EOS).
            return_outputs (bool): Whether to return logits and num_tokens. Defaults to False.
            label_smoothing (float): Label smoothing coefficient. Defaults to 0.0.
            src_probs (torch.Tensor, optional): Soft source probabilities for differentiable cycle.

        Returns:
            If return_outputs is False:
                tuple: (loss, num_tokens)
            If return_outputs is True:
                tuple: (loss, (logits, num_tokens))
        """
        # src: (batch, src_len)
        # tgt: (batch, tgt_len) - contains BOS and EOS

        # Create masks
        if src_probs is not None:
            src_padding_mask = (src_probs.argmax(dim=-1) == self.config.pad_id).to(torch.bool)
            device = src_probs.device
        else:
            src_padding_mask = (src == self.config.pad_id).to(torch.bool)
            device = src.device

        # For training, we align input and target
        # Input to decoder: tgt[:, :-1] (BOS ... last_token)
        # Target for loss: tgt[:, 1:] (first_token ... EOS)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_padding_mask = (tgt_input == self.config.pad_id).to(torch.bool)

        # 1. Encode
        memory = self.encode(src=src, src_probs=src_probs)

        # Causal mask for decoder autogression
        tgt_len = tgt_input.size(1)
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # 2. Decode using native causal flag AND explicit mask to fix torch.compile compatibility
        outs = self.decode(
            tgt_input,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # 3 & 4. Project and Loss
        mask = tgt_out != self.config.pad_id
        num_tokens = mask.sum()

        if return_outputs:
            logits = self.project(outs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size_tgt),
                tgt_out.reshape(-1),
                ignore_index=self.config.pad_id,
                label_smoothing=label_smoothing,
                reduction="sum",
            )
            return loss, (logits, num_tokens)
        else:
            # Optimal path for training: only project valid tokens
            outs_flat = outs[mask]
            logits_flat = self.project(outs_flat)
            tgt_out_flat = tgt_out[mask]

            loss = nn.functional.cross_entropy(
                logits_flat,
                tgt_out_flat,
                label_smoothing=label_smoothing,
                reduction="sum",
            )
            return loss, num_tokens

    def generate_gumbel(self, src, max_len=None, tau=1.0):
        """
        Differentiable greedy decoding using Gumbel-Softmax.
        """
        max_len = max_len or self.config.max_len
        bos_id = self.config.bos_id
        pad_id = self.config.pad_id
        
        bs = src.size(0)
        device = src.device
        
        memory = self.encode(src=src)
        src_padding_mask = (src == pad_id).to(torch.bool)
        
        ys_probs = torch.zeros(bs, 1, self.config.vocab_size_tgt, device=device)
        ys_probs[:, 0, bos_id] = 1.0
        
        for i in range(max_len):
            sz = ys_probs.size(1)
            tgt_mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
            
            out = self.decode(
                tgt=None, 
                memory=memory, 
                tgt_probs=ys_probs,
                tgt_mask=tgt_mask, 
                tgt_is_causal=True,
                memory_key_padding_mask=src_padding_mask
            )
            
            probs = self.project(out[:, -1])
            gumbel_prob = torch.nn.functional.gumbel_softmax(probs, tau=tau, hard=True).unsqueeze(1)
            ys_probs = torch.cat([ys_probs, gumbel_prob], dim=1)
            
        return ys_probs[:, 1:, :]

    @torch.no_grad()
    def generate(self, src, max_len=None, bos_id=None, eos_id=None, enc_output=None):
        """
        Greedy decoding for generation.

        Args:
            src (torch.Tensor): Source sequences.
            max_len (int, optional): Maximum generation length.
            bos_id (int, optional): Beginning-of-sentence token ID.
            eos_id (int, optional): End-of-sentence token ID.
            enc_output (torch.Tensor, optional): Pre-computed encoder output.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        max_len = max_len or self.config.max_len
        bos_id = bos_id if bos_id is not None else self.config.bos_id
        eos_id = eos_id if eos_id is not None else self.config.eos_id
        pad_id = self.config.pad_id

        src_padding_mask = (src == pad_id).to(torch.bool)
        bs = src.size(0)
        device = src.device

        if enc_output is None:
            memory = self.encode(src)
        else:
            memory = enc_output

        # Start with BOS
        ys = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(bs, dtype=torch.bool, device=device)

        for i in range(max_len):
            sz = ys.size(1)
            tgt_mask = torch.triu(
                torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1
            )
            out = self.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
                memory_key_padding_mask=src_padding_mask,
            )

            # Project last token
            prob = self.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Update sequences
            next_word = next_word.clone()
            next_word[finished] = pad_id
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Track finished
            finished = finished | (next_word == eos_id)

            if finished.all():
                break

        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, max_len=None, beam_size=5, bos_id=None, eos_id=None):
        """
        Beam search decoding for generation.

        Args:
            src (torch.Tensor): Source sequences.
            max_len (int, optional): Maximum generation length.
            beam_size (int): Number of beams. Defaults to 5.
            bos_id (int, optional): Beginning-of-sentence token ID.
            eos_id (int, optional): End-of-sentence token ID.

        Returns:
            torch.Tensor: Best sequence of token IDs for each input sequence.
        """
        max_len = max_len or self.config.max_len
        bos_id = bos_id if bos_id is not None else self.config.bos_id
        eos_id = eos_id if eos_id is not None else self.config.eos_id
        pad_id = self.config.pad_id

        # src: (bs, seq_len)
        bs = src.size(0)
        device = src.device

        # Encode once
        memory = self.encode(src)

        src_padding_mask = (src == pad_id).to(torch.bool)

        # Tile memory and mask
        memory = memory.repeat_interleave(beam_size, dim=0)
        src_padding_mask = src_padding_mask.repeat_interleave(beam_size, dim=0)

        # Initialize
        scores = torch.zeros(bs, beam_size, device=device)
        scores[:, 1:] = -1e9

        # inputs: (bs, beam_size, seq_len)
        inputs = torch.full((bs, beam_size, 1), bos_id, dtype=torch.long, device=device)

        vocab_size = self.config.vocab_size_tgt

        for i in range(max_len):
            curr_seq_len = inputs.size(2)
            flat_inputs = inputs.view(bs * beam_size, curr_seq_len)
            tgt_mask = torch.triu(
                torch.ones(curr_seq_len, curr_seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )

            # Decode
            out = self.decode(
                flat_inputs,
                memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
                memory_key_padding_mask=src_padding_mask,
            )

            # Logits for last token
            logits = self.project(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Reshape back to (bs, beam, vocab)
            log_probs = log_probs.view(bs, beam_size, vocab_size)

            # Add to previous scores
            total_scores = scores.unsqueeze(-1) + log_probs

            # Flatten to find top-k across all (beam * vocab) options
            total_scores_flat = total_scores.view(bs, -1)

            # Get top k
            top_acc_scores, top_indices = total_scores_flat.topk(beam_size, dim=-1)

            # Convert indices back
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update scores
            scores = top_acc_scores

            # Construct new inputs
            new_inputs = []
            for b in range(bs):
                prev_beams = inputs[b]
                selected_beam_indices = beam_indices[b]
                selected_tokens = token_indices[b]

                selected_sequences = prev_beams[selected_beam_indices]
                new_seq = torch.cat(
                    [selected_sequences, selected_tokens.unsqueeze(-1)], dim=-1
                )
                new_inputs.append(new_seq)

            inputs = torch.stack(new_inputs)

        # Return best beam
        return inputs[:, 0, 1:]  # Skip BOS


