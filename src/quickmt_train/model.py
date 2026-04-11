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
    ):
        super().__init__()
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
    ):
        super().__init__()
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
        if config.joint_vocab:
            # Share the same embedding for src and tgt
            self.tgt_tok_emb = self.src_tok_emb
        else:
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
        if config.joint_vocab:
            # With joint vocab, tie generator to the shared embedding
            self.generator.weight = self.src_tok_emb.embedding.weight
        elif config.tie_decoder_embeddings:
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
            # If weights are tied (joint_vocab or tie_decoder_embeddings),
            # the generator weight is just a pointer to embeddings.
            # We skip re-initializing it here to preserve the embedding stats.
            tied = self.config.joint_vocab or self.config.tie_decoder_embeddings
            if not (tied and module is self.generator):
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

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence of shape (batch_size, src_len).
            src_mask (torch.Tensor, optional): Encoder attention mask.

        Returns:
            torch.Tensor: Encoded sequence (memory).
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # Create padding mask: True where padding tokens exist
        src_padding_mask = (src == self.config.pad_id).to(torch.bool)

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

        Returns:
            torch.Tensor: Decoded features.
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

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

    def forward(self, src, tgt, return_outputs=False, label_smoothing=0.0):
        """
        Single training step: encode, decode, and calculate loss.

        Args:
            src (torch.Tensor): Source sequences.
            tgt (torch.Tensor): Target sequences (including BOS and EOS).
            return_outputs (bool): Whether to return logits and num_tokens. Defaults to False.
            label_smoothing (float): Label smoothing coefficient. Defaults to 0.0.

        Returns:
            If return_outputs is False:
                tuple: (loss, num_tokens)
            If return_outputs is True:
                tuple: (loss, (logits, num_tokens))
        """
        # src: (batch, src_len)
        # tgt: (batch, tgt_len) - contains BOS and EOS

        # Create masks
        src_padding_mask = (src == self.config.pad_id).to(torch.bool)

        # For training, we align input and target
        # Input to decoder: tgt[:, :-1] (BOS ... last_token)
        # Target for loss: tgt[:, 1:] (first_token ... EOS)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_padding_mask = (tgt_input == self.config.pad_id).to(torch.bool)

        # 1. Encode
        memory = self.encode(src)

        # 2. Decode using native causal flag (no explicit mask needed)
        outs = self.decode(
            tgt_input,
            memory,
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
            torch.Tensor: Generated token IDs (without BOS).
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
            # Use proper causal mask instead of torch.triu
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(device)
            out = self.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Project last token
            prob = self.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Update sequences - keep finished sequences unchanged
            next_word = next_word.clone()
            if finished.any():
                # Keep the last token of finished sequences as pad
                next_word[finished] = ys[finished, -1]
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Track finished sequences
            finished = finished | (next_word == eos_id)

            if finished.all():
                break

        # Remove BOS token from output
        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, max_len=None, beam_size=5, bos_id=None, eos_id=None,
                    length_penalty_alpha=1.0, length_penalty_beta=0.0):
        """
        Beam search decoding for generation with length normalization.

        Args:
            src (torch.Tensor): Source sequences.
            max_len (int, optional): Maximum generation length.
            beam_size (int): Number of beams. Defaults to 5.
            bos_id (int, optional): Beginning-of-sentence token ID.
            eos_id (int, optional): End-of-sentence token ID.
            length_penalty_alpha: Length normalization alpha. Defaults to 1.0.
            length_penalty_beta: Length normalization beta. Defaults to 0.0.

        Returns:
            torch.Tensor: Best sequence of token IDs for each input sequence.
        """
        max_len = max_len or self.config.max_len
        bos_id = bos_id if bos_id is not None else self.config.bos_id
        eos_id = eos_id if eos_id is not None else self.config.eos_id
        pad_id = self.config.pad_id

        bs = src.size(0)
        device = src.device

        # Encode once
        src_padding_mask = (src == pad_id).to(torch.bool)
        memory = self.encode(src)

        # Tile memory and mask for beam
        memory = memory.repeat_interleave(beam_size, dim=0)
        src_padding_mask = src_padding_mask.repeat_interleave(beam_size, dim=0)

        # Initialize: beam 0 gets BOS, others get dummy scores
        scores = torch.zeros(bs, beam_size, device=device)
        scores[:, 1:] = -1e9

        # Track which beams have finished
        # (batch, beam) -> whether beam finished
        finished_beams = torch.zeros(bs, beam_size, dtype=torch.bool, device=device)
        # (batch, beam) -> length of each beam
        beam_lengths = torch.ones(bs, beam_size, device=device)

        # inputs: (bs, beam_size, seq_len)
        inputs = torch.full((bs, beam_size, 1), bos_id, dtype=torch.long, device=device)

        vocab_size = self.config.vocab_size_tgt

        for step in range(max_len):
            curr_seq_len = inputs.size(2)
            flat_inputs = inputs.view(bs * beam_size, curr_seq_len)
            # Use proper causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_seq_len).to(device)

            # Decode
            out = self.decode(
                flat_inputs,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Logits for last token
            logits = self.project(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)
            # Mask out EOS for already-finished beams to prevent re-selection
            log_probs = log_probs.view(bs, beam_size, vocab_size)
            log_probs[finished_beams] = 0.0
            log_probs[finished_beams, :, eos_id] = -1e9
            log_probs = log_probs.view(bs * beam_size, vocab_size)

            # Reshape back to (bs, beam, vocab)
            log_probs = log_probs.view(bs, beam_size, vocab_size)

            # Apply length normalization to scores
            current_length = beam_lengths  # (bs, beam)
            length_penalty = ((length_penalty_beta + current_length) ** length_penalty_alpha) / (
                (length_penalty_beta + 1.0) ** length_penalty_alpha
            )
            normalized_scores = scores / length_penalty.clamp(min=1e-6)

            # Add to previous scores
            total_scores = normalized_scores.unsqueeze(-1) + log_probs

            # Flatten to find top-k
            total_scores_flat = total_scores.view(bs, -1)

            # Get top k
            top_scores, top_indices = total_scores_flat.topk(beam_size, dim=-1)

            # Convert indices back
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update scores
            scores = top_scores

            # Construct new inputs and track finished beams
            new_inputs_list = []
            new_finished = []
            new_lengths = []

            for b in range(bs):
                prev_beams = inputs[b]
                selected_beam_idx = beam_indices[b]
                selected_tokens = token_indices[b]

                selected_sequences = prev_beams[selected_beam_idx]
                new_tokens = selected_tokens.unsqueeze(-1)
                new_seq = torch.cat([selected_sequences, new_tokens], dim=-1)

                # Track finished beams
                prev_finished = finished_beams[b, selected_beam_idx]
                just_finished = (selected_tokens == eos_id) & (~prev_finished)
                curr_finished = prev_finished | just_finished

                # Update lengths
                curr_lengths = beam_lengths[b, selected_beam_idx] + (~curr_finished).long()

                new_inputs_list.append(new_seq)
                new_finished.append(curr_finished)
                new_lengths.append(curr_lengths)

            inputs = torch.stack(new_inputs_list)
            finished_beams = torch.stack(new_finished)
            beam_lengths = torch.stack(new_lengths)

            # Early exit: if all beams in all batches have finished
            if finished_beams.all():
                break

        # Apply final length normalization and select best beam
        final_length_penalty = ((length_penalty_beta + beam_lengths) ** length_penalty_alpha) / (
            (length_penalty_beta + 1.0) ** length_penalty_alpha
        )
        final_scores = scores / final_length_penalty.clamp(min=1e-6)

        # Select best beam for each batch
        best_beam_idx = final_scores.argmax(dim=1)
        best_beams = []
        for b in range(bs):
            beam = inputs[b, best_beam_idx[b].item()]
            # Remove BOS token
            best_beams.append(beam[1:])

        return torch.stack(best_beams)

    def convert_to_int8(self):
        """
        Convert the model to a quantized INT8 model using static quantization (PTQ).
        Should be called after calibration with calibrate().
        """
        self.eval()
        torch.ao.quantization.convert(self, inplace=True)
        print("Model converted to INT8")

    def calibrate(self, loader, num_batches=10):
        """
        Update quantization observers by running sample data through the model.
        Useful for Post-Training Quantization (PTQ) after training or weight averaging.

        Args:
            loader (DataLoader): Data loader providing (src, tgt) pairs.
            num_batches (int): Number of batches to use for calibration. Defaults to 10.
        """
        device = next(self.parameters()).device
        self.eval()
        # Ensure observers are enabled and Dropout is disabled
        with torch.no_grad():
            for i, (src, tgt) in enumerate(loader):
                if i >= num_batches:
                    break
                # Run forward pass (updates observers)
                self.forward(src.to(device), tgt.to(device))
        print(f"Calibration completed on {num_batches} batches")
