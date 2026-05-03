from dataclasses import dataclass, fields
from enum import Enum
import os


class StrEnum(str, Enum):
    pass


class ActivationType(StrEnum):
    GELU = "gelu"
    RELU = "relu"
    SWIGLU = "swiglu"
    SILU = "silu"


class MLPType(StrEnum):
    STANDARD = "standard"
    GATED = "gated"


class NormType(StrEnum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class SchedulerType(StrEnum):
    INV_SQRT = "inv_sqrt"
    COSINE = "cosine"


class DeviceType(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


class PrecisionType(StrEnum):
    BF16 = "bf16"
    BFLOAT16 = "bfloat16"
    FP16 = "fp16"
    FLOAT16 = "float16"
    FP32 = "fp32"
    FLOAT32 = "float32"


class CheckpointStrategy(StrEnum):
    RECENT = "recent"
    BEST = "best"


class EarlyStoppingMetric(StrEnum):
    LOSS = "loss"
    PPL = "ppl"
    ACC = "acc"
    BLEU = "bleu"
    CHRF = "chrf"

    @property
    def lower_is_better(self) -> bool:
        return self in (EarlyStoppingMetric.LOSS, EarlyStoppingMetric.PPL)


class QuantizationType(StrEnum):
    INT8 = "int8"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    NONE = "none"


class TokenizerType(StrEnum):
    UNIGRAM = "unigram"
    BPE = "bpe"


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture."""

    d_model: int = 768
    enc_layers: int = 12
    dec_layers: int = 2
    n_heads: int = 16
    n_kv_heads: int = None
    ffn_dim: int = 4096
    max_len: int = 512  # Hard filter during data loading
    dropout: float = 0.1
    vocab_size_src: int = 32000
    vocab_size_tgt: int = 32000
    use_checkpoint: bool = False
    ff_bias: bool = True
    layernorm_eps: float = 1e-6
    activation: ActivationType = ActivationType.GELU
    mlp_type: MLPType = MLPType.GATED
    norm_type: NormType = NormType.RMSNORM
    tie_decoder_embeddings: bool = False
    joint_vocab: bool = False

    # Special Tokens
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


@dataclass
class CorpusConfig:
    src_file: str
    tgt_file: str
    weight: int = 1
    start_step: int = 0
    stop_step: int = 1000000000


@dataclass
class DataConfig:
    """Configuration for data loading, preprocessing, and tokenization."""

    # Experiment info (usually populated from TrainConfig)
    experiment_name: str = "default"

    # Languages
    src_lang: str = "fa"
    tgt_lang: str = "en"

    # Paths
    corpora: list["CorpusConfig"] = None  # Will be initialized in __post_init__
    src_dev_path: str = "data/dev.fa"
    tgt_dev_path: str = "data/dev.en"

    def __post_init__(self):
        if self.corpora is None:
            self.corpora = []

    # Tokenizer
    char_coverage: float = 0.9999
    input_sentence_size: int = 10_000_000
    train_joint_tokenizer: bool = False
    tokenizer_type: TokenizerType = TokenizerType.UNIGRAM

    @property
    def tokenizer_prefix_src(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_src")

    @property
    def tokenizer_prefix_tgt(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_tgt")

    # Streaming & Batching
    max_tokens_per_batch: int = 6000
    buffer_size: int = 40000
    num_workers: int = 4
    prefetch_factor: int = 128
    pad_multiple: int = 1

    # N-best sampling
    src_spm_nbest_size: int = 1
    tgt_spm_nbest_size: int = 1
    src_spm_alpha: float = 0.0
    tgt_spm_alpha: float = 0.0


@dataclass
class TrainConfig:
    """Configuration for the training loop and optimization."""

    experiment_name: str = "default"
    aim_repo: str = "./aim-runs"

    # Optimizer
    lr: float = 1.0e-3
    weight_decay: float = 0.01
    weight_decay_embeddings: bool = False
    adam_eps: float = 1e-6
    label_smoothing: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.998
    z_loss_coeff: float = 1e-4

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_start_step: int = 10000

    # Scheduler
    scheduler_type: SchedulerType = SchedulerType.INV_SQRT
    warmup_steps: int = 5000
    max_steps: int = 100000
    epochs: int = 20

    # Training Loop
    accum_steps: int = 30
    grad_clip: float = 1.0
    eval_steps: int = 1000
    max_checkpoints: int = 10
    save_checkpoints: bool = True
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.BEST

    # Early Stopping
    early_stopping_patience: int = 0
    early_stopping_metric: EarlyStoppingMetric = EarlyStoppingMetric.CHRF

    # Hardware & Performance
    device: DeviceType = DeviceType.CUDA
    precision: PrecisionType = PrecisionType.BF16
    tf32: bool = True

    # Logging & Validation
    log_steps: int = 100
    val_max_samples: int = 500
    quick_test_samples: int = 5

    # Checkpoint Resume
    resume_from: str = ""  # Path to .pt or .safetensors checkpoint
    reset_optimizer: bool = False  # Reset optimizer/scheduler state (for fine-tuning)

    # torch_compile
    enable_torch_compile: bool = True

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_name, "checkpoints")


@dataclass
class ExportConfig:
    """Configuration for checkpoint averaging, quantization, and export."""

    # Averaging
    k: int = 5
    ignore_ema: bool = False

    # Quantization
    quantization: QuantizationType = QuantizationType.BFLOAT16

    # Inference Defaults
    beam_size: int = 5
    max_len: int = 512
    batch_size: int = 32

    # CT2 specific
    add_source_bos: bool = True
    add_source_eos: bool = False

    # Experiment info (usually populated from TrainConfig)
    experiment_name: str = "default"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.experiment_name, "exported_model")

    @property
    def src_vocab(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_src.vocab")

    @property
    def tgt_vocab(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_tgt.vocab")

    @property
    def output_prefix(self) -> str:
        return os.path.join(self.experiment_name, "averaged_model")


def _from_dict(cls, d):
    valid_fields = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in valid_fields:
            raise ValueError(f"Invalid key '{k}' for config {cls.__name__}")

        field_type = valid_fields[k]
        if k == "corpora" and isinstance(v, list):
            kwargs[k] = [CorpusConfig(**c) if isinstance(c, dict) else c for c in v]
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            try:
                kwargs[k] = field_type(v)
            except ValueError:
                valid_vals = [e.value for e in field_type]
                raise ValueError(
                    f"Invalid value '{v}' for {k} in {cls.__name__}. Expected one of {valid_vals}"
                )
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str):
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    valid_top_level = {"model", "data", "train", "export"}
    for k in cfg.keys():
        if k not in valid_top_level:
            raise ValueError(
                f"Invalid top-level section '{k}' in config file. Expected one of {valid_top_level}"
            )

    model_config = _from_dict(ModelConfig, cfg.get("model", {}))
    data_config = _from_dict(DataConfig, cfg.get("data", {}))
    train_config = _from_dict(TrainConfig, cfg.get("train", {}))
    export_config = _from_dict(ExportConfig, cfg.get("export", {}))

    # Link experiment name across configs
    export_config.experiment_name = train_config.experiment_name
    data_config.experiment_name = train_config.experiment_name

    return model_config, data_config, train_config, export_config
