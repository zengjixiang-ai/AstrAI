"""Storage writer strategies for pipeline output.

The :class:`StoreWriter` abstraction decouples the pipeline from the
concrete storage format (bin / h5).  The pipeline builds a ``{key:
List[Tensor]}`` dict and delegates the write to the writer selected
by ``output.storage_format``.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from astrai.dataset.storage import save_bin, save_h5
from astrai.factory import BaseFactory


class StoreWriter(ABC):
    """Write pre-tokenized tensors to disk in a format-specific way."""

    @abstractmethod
    def save(
        self,
        output_dir: str,
        domain: str,
        shard_idx: int,
        tensors: Dict[str, List[torch.Tensor]],
    ) -> None: ...


class StoreWriterFactory(BaseFactory["StoreWriter"]):
    @classmethod
    def _validate_component(cls, component_cls: type):
        if not issubclass(component_cls, StoreWriter):
            raise TypeError(f"{component_cls.__name__} must inherit from StoreWriter")


@StoreWriterFactory.register("bin")
class BinWriter(StoreWriter):
    def save(self, output_dir, domain, shard_idx, tensors):
        shard_path = os.path.join(output_dir, domain, f"shard_{shard_idx:04d}")
        save_bin(shard_path, tensors)


@StoreWriterFactory.register("h5")
class H5Writer(StoreWriter):
    def save(self, output_dir, domain, shard_idx, tensors):
        chunk_dir = os.path.join(output_dir, domain)
        save_h5(chunk_dir, f"data_{shard_idx:04d}", tensors)
