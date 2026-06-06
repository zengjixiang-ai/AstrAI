"""Position-id generation strategies for packed sequences.

Each strategy takes the list of per-document token sequences after packing
and returns a flat list of position ids (same total length as all
sequences combined).  The pipeline wraps the result into a tensor and
attaches it as ``position_ids``.
"""

from abc import ABC, abstractmethod
from typing import List

from astrai.factory import BaseFactory


class PositionIdStrategy(ABC):
    """Generate ``position_ids`` for packed sequences."""

    @abstractmethod
    def generate(self, sequences: List[list]) -> List[int]: ...


class PositionIdStrategyFactory(BaseFactory["PositionIdStrategy"]):
    pass


@PositionIdStrategyFactory.register("none")
class NoPositionId(PositionIdStrategy):
    def generate(self, sequences):
        return []


@PositionIdStrategyFactory.register("doc_reset")
class DocResetPositionId(PositionIdStrategy):
    def generate(self, sequences):
        pos_ids = []
        for seq in sequences:
            pos_ids.extend(range(len(seq)))
        return pos_ids


@PositionIdStrategyFactory.register("continuous")
class ContinuousPositionId(PositionIdStrategy):
    def generate(self, sequences):
        total = sum(len(seq) for seq in sequences)
        return list(range(total))
