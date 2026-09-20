from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .environment import ContinuousEnvironment
from .symbolic import FrontierTransition, SymbolicBlock, SymbolicPartitioner


class FrontierSupportError(RuntimeError):
    """An executable first-exit transition violates exact frontier support."""


@dataclass(frozen=True)
class AbsorptionResult:
    """某个符号块在固定动作下到达前沿前的吸收量。"""

    block_id: int
    action: float
    expected_cost: float
    expected_time: float
    hit_distribution: Mapping[int, float]
    samples: int
    frontier_hits: int
    model_calls: int


class AbsorptionAnalyzer:
    """用连续抽样近似内部块到符号前沿的首次到达量。"""

    def __init__(
        self,
        env: ContinuousEnvironment,
        partitioner: SymbolicPartitioner,
        blocks: Sequence[SymbolicBlock],
        *,
        frontier_transitions: Sequence[FrontierTransition] = (),
        frontier_backend: str = "sampled-fallback",
        samples_per_block: int = 48,
        max_steps: int = 80,
        truncate_edges: bool = True,
        seed: int = 11,
    ) -> None:
        self.env = env
        self.partitioner = partitioner
        self.blocks = tuple(blocks)
        self.samples_per_block = samples_per_block
        self.max_steps = max_steps
        self.truncate_edges = truncate_edges
        self.rng = np.random.default_rng(seed)
        self.frontier_backend = frontier_backend
        self.strict_frontier_support = frontier_backend == "exact-lra"
        self._blocks_by_id = {block.block_id: block for block in self.blocks}
        self._frontier_support: dict[tuple[int, float], set[int]] = {}
        for transition in frontier_transitions:
            key = (transition.source_block, float(transition.action))
            self._frontier_support.setdefault(key, set()).add(transition.target_block)
        self._observed_frontier_keys: set[tuple[int, float, int]] = set()
        self._cache: dict[tuple[int, float], AbsorptionResult] = {}

    @property
    def observed_frontier_transitions(self) -> tuple[FrontierTransition, ...]:
        """Return first-exit edges observed during absorption rollouts."""

        transitions: list[FrontierTransition] = []
        for source_id, action, target_id in sorted(self._observed_frontier_keys):
            transitions.append(
                self.partitioner.make_frontier_transition(
                    self._blocks_by_id[source_id],
                    action,
                    self._blocks_by_id[target_id],
                    origin="absorption-rollout",
                )
            )
        return tuple(transitions)

    def analyze_all(self) -> dict[tuple[int, float], AbsorptionResult]:
        for block in self.blocks:
            for action in self.env.actions:
                self.analyze(block, float(action))
        return dict(self._cache)

    def analyze(self, block: SymbolicBlock, action: float) -> AbsorptionResult:
        key = (block.block_id, float(action))
        if key in self._cache:
            return self._cache[key]

        if self.truncate_edges and block.is_truncated:
            result = self._truncated_result(block, float(action))
            self._cache[key] = result
            return result

        starts = self.partitioner.sample_block(block, self.samples_per_block)
        costs: list[float] = []
        times: list[float] = []
        hits: dict[int, int] = {}
        frontier_hits = 0
        model_calls = 0

        for start in starts:
            cost, duration, target_id, hit_frontier, rollout_calls = self._rollout_until_exit(
                block.block_id,
                start,
                float(action),
            )
            costs.append(cost)
            times.append(duration)
            hits[target_id] = hits.get(target_id, 0) + 1
            frontier_hits += int(hit_frontier)
            model_calls += rollout_calls

        total = max(len(starts), 1)
        hit_distribution = {block_id: count / total for block_id, count in hits.items()}
        result = AbsorptionResult(
            block_id=block.block_id,
            action=float(action),
            expected_cost=float(np.mean(costs)) if costs else 0.0,
            expected_time=max(float(np.mean(times)) if times else self.env.dt, 1e-9),
            hit_distribution=hit_distribution,
            samples=total,
            frontier_hits=frontier_hits,
            model_calls=model_calls,
        )
        self._cache[key] = result
        return result

    def _truncated_result(self, block: SymbolicBlock, action: float) -> AbsorptionResult:
        safety = self.env.safety_action()
        penalty = 100.0 if action != safety else 20.0
        return AbsorptionResult(
            block_id=block.block_id,
            action=float(action),
            expected_cost=penalty,
            expected_time=self.env.dt,
            hit_distribution={block.block_id: 1.0},
            samples=0,
            frontier_hits=0,
            model_calls=0,
        )

    def _rollout_until_exit(
        self,
        start_block_id: int,
        start_state: np.ndarray,
        action: float,
    ) -> tuple[float, float, int, bool, int]:
        state = np.asarray(start_state, dtype=float)
        total_cost = 0.0
        total_time = 0.0
        target_id = start_block_id
        model_calls = 0

        for _ in range(self.max_steps):
            step = self.env.step(state, action)
            model_calls += 1
            total_cost += step.cost
            total_time += float(step.info.get("transition_time", self.env.dt))
            target = self.partitioner.block_for_state(step.next_state, self.blocks)
            if target is None:
                return total_cost, max(total_time, self.env.dt), start_block_id, False, model_calls
            target_id = target.block_id
            if step.done or target_id != start_block_id:
                if target_id != start_block_id:
                    self._record_frontier_hit(start_block_id, action, target_id)
                return total_cost, max(total_time, self.env.dt), target_id, target_id != start_block_id, model_calls
            state = step.next_state

        return total_cost, max(total_time, self.env.dt), target_id, False, model_calls

    def _record_frontier_hit(self, source_id: int, action: float, target_id: int) -> None:
        key = (source_id, float(action))
        if self.strict_frontier_support and target_id not in self._frontier_support.get(key, set()):
            raise FrontierSupportError(
                "Absorption rollout observed a first-exit transition outside the exact-LRA frontier support: "
                f"B{source_id} --a={float(action):g}--> B{target_id}."
            )
        self._observed_frontier_keys.add((source_id, float(action), target_id))
