from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DiscreteActionSpec:
    """有限动作空间。"""

    actions: np.ndarray

    @property
    def continuous(self) -> bool:
        return False


@dataclass(frozen=True)
class ContinuousActionSpec:
    """连续动作空间及其规划候选集参数。"""

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    critical_values: tuple[float, ...] = ()
    max_candidates: int = 17
    refinement_rounds: int = 2

    @property
    def continuous(self) -> bool:
        return True

    def clip_scalar(self, action: float | Sequence[float] | np.ndarray) -> float:
        value = float(np.asarray(action, dtype=float).reshape(-1)[0])
        return float(np.clip(value, float(self.low[0]), float(self.high[0])))

    def initial_candidates(self) -> np.ndarray:
        """生成边界、零点、中点和符号临界动作组成的初始候选集。"""

        low, high = float(self.low[0]), float(self.high[0])
        midpoint = 0.5 * (low + high)
        values = [low, 0.0, midpoint, high, *self.critical_values]
        return _unique_sorted_clipped(values, low, high, self.max_candidates)

    def refine_candidates(self, best_action: float, rounds: int | None = None) -> np.ndarray:
        """围绕当前最优动作做局部细化，供执行阶段滚动优化使用。"""

        low, high = float(self.low[0]), float(self.high[0])
        values = list(self.initial_candidates())
        radius = 0.25 * (high - low)
        for _ in range(self.refinement_rounds if rounds is None else rounds):
            values.extend([best_action - radius, best_action, best_action + radius])
            radius *= 0.5
        return _unique_sorted_clipped(values, low, high, self.max_candidates)


def cartesian_candidates(per_dimension: Sequence[Sequence[float]], max_candidates: int = 81) -> np.ndarray:
    """为将来的多维连续动作生成有上限的笛卡尔积候选集。"""

    rows = np.asarray(list(product(*per_dimension)), dtype=float)
    if len(rows) <= max_candidates:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_candidates, dtype=int)
    return rows[indices]


def _unique_sorted_clipped(values: Sequence[float], low: float, high: float, limit: int) -> np.ndarray:
    clipped = sorted({float(np.clip(round(float(np.clip(value, low, high)), 12), low, high)) for value in values})
    if len(clipped) <= limit:
        return np.asarray(clipped, dtype=float)
    indices = np.linspace(0, len(clipped) - 1, limit, dtype=int)
    return np.asarray([clipped[index] for index in indices], dtype=float)
