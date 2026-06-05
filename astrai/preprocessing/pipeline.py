"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
sharding and flush to ``.h5`` / ``.bin`` storage.
"""

import json
import os
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Tuple

import torch
import tqdm

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset.storage import save_bin, save_h5
from astrai.preprocessing.builder import SectionedMaskBuilder
from astrai.tokenize import AutoTokenizer

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def filter_by_length(text: str, min_len: int = 50, max_len: int = 2_000_000) -> bool:
    return min_len <= len(text) <= max_len


def _truncate(seq: list, max_len: int, mode: str) -> list:
    if len(seq) <= max_len:
        return seq
    if mode == "keep_end":
        return seq[-max_len:]
    return seq[:max_len]


def pack_sequences(
    sequences: List[list],
    max_packed_len: int,
    strategy: str,
    truncation_mode: str,
) -> List[Tuple[int, int]]:
    """Pack *sequences* into bins and return a reorder plan.

    Returns a list of ``(orig_idx, truncated_length)`` in flush order.
    All keys (sequence, loss_mask, …) must be reordered and truncated
    identically according to this plan.

    Supported *strategy* values:

    - ``"simple"``: sequential, no reordering.
    - ``"bfd"``: best-fit decreasing bin packing.
    """
    n = len(sequences)
    if strategy == "simple":
        return [(i, min(len(sequences[i]), max_packed_len)) for i in range(n)]

    order = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)
    bins: List[List[int]] = []
    bin_lengths: List[int] = []

    for orig_idx in order:
        seq_len = min(len(sequences[orig_idx]), max_packed_len)

        best_bin = None
        best_remain = max_packed_len + 1
        for i, bl in enumerate(bin_lengths):
            remain = max_packed_len - bl
            if seq_len <= remain < best_remain:
                best_remain = remain
                best_bin = i

        if best_bin is not None:
            bins[best_bin].append(orig_idx)
            bin_lengths[best_bin] += seq_len
        else:
            bins.append([orig_idx])
            bin_lengths.append(seq_len)

    plan: List[Tuple[int, int]] = []
    for bin_indices in bins:
        for orig_idx in bin_indices:
            plan.append((orig_idx, min(len(sequences[orig_idx]), max_packed_len)))

    return plan


class Pipeline:
    """Tokenization pipeline driven by a declarative :class:`PipelineConfig`.

    Usage::

        config = PipelineConfig.from_json("sft_pipeline.json")
        Pipeline(config, ["data.jsonl"], output_dir="out", tokenizer_path="params").run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        input_paths: list[str],
        output_dir: str,
        tokenizer_path: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.config = config
        self.paths = input_paths
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path

        self.mask_builder = SectionedMaskBuilder()

    def transform(self, item: dict) -> Optional[dict]:
        return self.mask_builder.build(item, self.config, self._tokenizer)

    def run(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        domains: dict = defaultdict(lambda: defaultdict(list))
        total_tokens = 0
        shard_idx: dict[str, int] = defaultdict(int)
        count = 0

        pp = self.config.preprocessing

        for item in tqdm.tqdm(
            self._iter_items(), desc="Tokenizing", unit="docs", mininterval=0.5
        ):
            if pp.max_items and count >= pp.max_items:
                break

            result = self.transform(item)
            if result is None:
                continue

            domain = result.pop("domain", "__default__")

            is_multi = bool(getattr(self.config.input, "sources", None))
            if is_multi:
                ids = self._primary_ids(result)
            else:
                ids = result.pop("sequence")
                result["sequence"] = ids

            if not ids:
                continue

            bucket = domains[domain]
            self._align_bucket(bucket, result, ids, is_multi)
            for key, val in result.items():
                bucket[key].append(val)

            count += 1
            total_tokens += len(ids)

            if total_tokens >= self.config.output.max_tokens_per_shard:
                self._flush(domains, shard_idx)
                domains.clear()
                total_tokens = 0

        if total_tokens > 0:
            self._flush(domains, shard_idx)

        print(f"Done. {count} documents tokenized.")

    @staticmethod
    def _primary_ids(result: dict) -> list:
        """Return the first list-valued entry in *result* as the primary id
        sequence for token counting."""
        for val in result.values():
            if isinstance(val, list) and val and isinstance(val[0], int):
                return val
        return []

    @staticmethod
    def _align_bucket(bucket: dict, result: dict, ids: list, is_multi: bool):
        """Pad previously-accumulated keys that are missing from *result*."""
        for key in list(bucket.keys()):
            if key in result:
                continue
            if is_multi:
                pad = bucket[key][-1] if bucket[key] else [1] * len(ids)
                bucket[key].append(pad)
            else:
                bucket[key].append([1] * len(ids))

    def _iter_items(self):
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

    def _flush(self, domains, shard_idx):
        for domain, keys in domains.items():
            idx = shard_idx[domain]
            chunk_dir = os.path.join(self.output_dir, domain)

            pp = self.config.preprocessing
            if pp.packing_strategy != "simple" and "sequence" in keys:
                plan = pack_sequences(
                    keys["sequence"],
                    pp.max_packed_len,
                    pp.packing_strategy,
                    pp.truncation_mode,
                )
                reordered = defaultdict(list)
                for orig_idx, truncated_len in plan:
                    for k, vals in keys.items():
                        reordered[k].append(
                            _truncate(
                                vals[orig_idx], pp.max_packed_len, pp.truncation_mode
                            )
                        )
                keys = reordered

            tensors = {}
            for key, ids_list in keys.items():
                dt = _STR_TO_DTYPE.get(
                    self.config.output.dtype.get(key, "int32"), torch.int32
                )
                tensors[key] = [
                    torch.tensor(list(chain.from_iterable(ids_list)), dtype=dt)
                ]

            pid_mode = self.config.output.position_ids_mode
            if pid_mode and pid_mode != "none" and "sequence" in tensors:
                pos_ids = []
                if pid_mode == "doc_reset":
                    for item in keys["sequence"]:
                        pos_ids.extend(range(len(item)))
                else:
                    total = sum(len(item) for item in keys["sequence"])
                    pos_ids = list(range(total))
                tensors["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

            shard_path = os.path.join(chunk_dir, f"shard_{idx:04d}")
            fmt = self.config.output.storage_format
            if fmt == "bin":
                save_bin(shard_path, tensors)
            else:
                save_h5(chunk_dir, f"data_{idx:04d}", tensors)
            shard_idx[domain] = idx + 1
            first_key = "sequence" if "sequence" in tensors else next(iter(tensors))
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({tensors[first_key][0].numel():,} tokens)"
            )
