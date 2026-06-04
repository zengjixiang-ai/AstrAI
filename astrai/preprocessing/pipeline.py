"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
sharding and flush to ``.h5`` / ``.bin`` storage.
"""

import json
import os
from collections import defaultdict
from itertools import chain
from typing import Optional

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
