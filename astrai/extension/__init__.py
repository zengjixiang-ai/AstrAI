import importlib
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_available: dict[str, bool] = {}
_modules: dict[str, object] = {}

for _name in ["gqa_decode_attn", "gqa_prefill_attn"]:
    try:
        _mod = importlib.import_module(f".{_name}", package=__package__)
        _available[_name] = True
        _modules[_name] = _mod
    except ImportError:
        _available[_name] = False
        _modules[_name] = None


def _expand_kv_heads(
    k: torch.Tensor, v: torch.Tensor, q_head: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V heads to match Q heads for GQA fallback."""
    kv_head = k.size(1)
    if kv_head == q_head:
        return k, v
    group = q_head // kv_head
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    return k, v


def _torch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    k, v = _expand_kv_heads(k, v, q.size(1))
    attn_mask = mask[:, None, None, :] if mask is not None else None
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal and mask is None, scale=scale
    )


def gqa_decode_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    causal_offset: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    if _available["gqa_decode_attn"]:
        return _modules["gqa_decode_attn"].gqa_decode_attn(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            causal_offset=causal_offset,
            scale=scale,
        )
    return _torch_fallback(q, k, v, mask, is_causal, scale)


def gqa_prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    causal_offset: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    if _available["gqa_prefill_attn"]:
        return _modules["gqa_prefill_attn"].gqa_prefill_attn(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            causal_offset=causal_offset,
            scale=scale,
        )
    return _torch_fallback(q, k, v, mask, is_causal, scale)
