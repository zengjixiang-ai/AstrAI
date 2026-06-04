"""Pipeline configuration for JSONL preprocessing.

Supports single-sequence (SFT/pretrain) and multi-output (DPO/GRPO)
modes, both driven declaratively through ``input.sections`` or
``input.sources``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from astrai.config.base import BaseConfig


@dataclass
class InputConfig(BaseConfig):
    """Declarative input mapping.

    Single-output mode (backward-compatible)::

        {"input": {"sections": [{"field": "messages", ...}]}}

    Multi-output mode (DPO / GRPO)::

        {"input": {"sources": {
            "chosen": {"sections": [{"field": "chosen", ...}]},
            "rejected": {"sections": [{"field": "rejected", ...}]},
        }}}
    """

    sections: Optional[List[Dict]] = None
    sources: Optional[Dict[str, Dict]] = None


@dataclass
class ProcessingConfig(BaseConfig):
    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    max_items: Optional[int] = None


@dataclass
class OutputConfig(BaseConfig):
    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000
    dtype: Dict[str, str] = field(default_factory=dict)
    position_ids_mode: Optional[str] = None
    """How to compute position_ids in packed sequences.

    - ``None`` / ``"none"``: do not generate (backward compatible).
    - ``"doc_reset"``: reset to 0 at each document boundary.
    - ``"continuous"``: sequential 0, 1, 2, ... (pretrain, single doc).
    """


@dataclass
class PipelineConfig(BaseConfig):
    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
