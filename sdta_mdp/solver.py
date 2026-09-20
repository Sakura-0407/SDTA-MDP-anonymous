from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from .absorption import AbsorptionAnalyzer, AbsorptionResult, FrontierSupportError
from .environment import ContinuousEnvironment
from .symbolic import FrontierTransition, PartitionReport, SymbolicBlock, SymbolicPartitioner


@dataclass(frozen=True)
class SMDPArrays:
    """嵌入在有限符号前沿集合上的 SMDP 数组表示。"""

    transitions: np.ndarray
    costs: np.ndarray
    times: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class SolverResult:
    """SDTA-MDP 求解输出。"""

    eta: float
    frontier_values: np.ndarray
    block_policy: Mapping[int, float]
    blocks: tuple[SymbolicBlock, ...]
    partition_report: PartitionReport
    absorption_results: Mapping[tuple[int, float], AbsorptionResult]
    iterations: int
    seconds: float
    partition_seconds: float
    absorption_seconds: float
    frontier_solve_seconds: float
    model_calls: int

    def action_for_block(self, block_id: int) -> float:
        return float(self.block_policy[block_id])


class DTPTASolver:
    """Symbolic Distributed Time-Aggregated MDP 主求解器。"""

    def __init__(
        self,
        env: ContinuousEnvironment,
        *,
        absorption_samples: int = 48,
        absorption_horizon: int = 80,
        tolerance: float = 1e-7,
        max_iter: int = 500,
        frontier_probe_samples: int = 8,
        frontier_mode: str = "auto",
        truncate_edges: bool = True,
        support_aware_control: bool = False,
        seed: int = 17,
    ) -> None:
        self.env = env
        self.absorption_samples = absorption_samples
        self.absorption_horizon = absorption_horizon
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.truncate_edges = truncate_edges
        self.support_aware_control = bool(support_aware_control)
        self.seed = seed
        self.partitioner = SymbolicPartitioner(
            env,
            frontier_probe_samples=frontier_probe_samples,
            frontier_mode=frontier_mode,
            seed=seed,
        )

    def solve(self) -> SolverResult:
        start = perf_counter()
        partition_start = perf_counter()
        report = self.partitioner.discover()
        partition_seconds = perf_counter() - partition_start
        analyzer = AbsorptionAnalyzer(
            self.env,
            self.partitioner,
            report.blocks,
            frontier_transitions=report.frontier_transitions,
            frontier_backend=report.frontier_backend,
            samples_per_block=self.absorption_samples,
            max_steps=self.absorption_horizon,
            truncate_edges=self.truncate_edges,
            seed=self.seed + 1,
        )
        absorption_start = perf_counter()
        absorption = analyzer.analyze_all()
        absorption_seconds = perf_counter() - absorption_start
        if report.frontier_backend != "exact-lra":
            report = replace(
                report,
                frontier_transitions=analyzer.observed_frontier_transitions,
            )
        solve_start = perf_counter()
        smdp = self._build_smdp(
            report.blocks,
            absorption,
            report.frontier_transitions,
        )
        policy_idx, eta, values, iterations = self._frontier_policy_iteration(smdp)
        frontier_solve_seconds = perf_counter() - solve_start
        block_policy = {
            block.block_id: float(self.env.actions[policy_idx[block.block_id]])
            for block in report.blocks
        }
        return SolverResult(
            eta=eta,
            frontier_values=values,
            block_policy=block_policy,
            blocks=report.blocks,
            partition_report=report,
            absorption_results=absorption,
            iterations=iterations,
            seconds=perf_counter() - start,
            partition_seconds=partition_seconds,
            absorption_seconds=absorption_seconds,
            frontier_solve_seconds=frontier_solve_seconds,
            model_calls=sum(item.model_calls for item in absorption.values()),
        )

    def make_policy(self, result: SolverResult) -> Callable[[np.ndarray], float]:
        def policy(state: np.ndarray) -> float:
            block = self.partitioner.block_for_state(state, result.blocks)
            if block is None:
                return self.env.safety_action()
            if self.support_aware_control:
                return self._support_aware_policy_action(
                    state,
                    block,
                    result.action_for_block(block.block_id),
                )
            return self.env.local_policy_action(state, result.action_for_block(block.block_id))

        return policy

    def _support_aware_policy_action(
        self,
        state: np.ndarray,
        block: SymbolicBlock,
        fallback_action: float,
    ) -> float:
        state_arr = self.env.bounds.clip(state)
        candidates = np.asarray(
            self.env.local_action_candidates(state_arr, fallback_action),
            dtype=float,
        )
        unsafe_paths = set(self.env.unsafe_path_names())
        if not unsafe_paths:
            return self.env.local_policy_action(state_arr, fallback_action)
        safe_candidates = [
            float(action)
            for action in candidates
            if unsafe_paths.isdisjoint(self._path_support_for_action(block, float(action)))
        ]
        if safe_candidates:
            candidates = np.asarray(safe_candidates, dtype=float)

        best_action = float(candidates[0])
        best_score = float("inf")
        for action in candidates:
            score = self._stochastic_first_step_score(
                state_arr,
                float(action),
                self.env.local_policy_depth(),
            )
            if score < best_score:
                best_score = score
                best_action = float(action)
        return best_action

    @staticmethod
    def _path_support_for_action(block: SymbolicBlock, action: float) -> tuple[str, ...]:
        if not block.action_paths:
            return ()
        representative = min(block.action_paths, key=lambda item: abs(float(item) - action))
        return tuple(block.action_paths[representative])

    def _stochastic_first_step_score(self, state: np.ndarray, action: float, depth: int) -> float:
        outcome_fn = getattr(self.env, "expected_step_outcomes", None)
        if callable(outcome_fn):
            outcomes = outcome_fn(state, action, quadrature_order=7)
        else:
            outcomes = ((1.0, self.env.step(state, action)),)
        base_env = getattr(self.env, "base_env", self.env)
        score = 0.0
        for probability, step in outcomes:
            penalty = 100.0 if step.info.get("constraint_violation", False) else 0.0
            if depth <= 1 or step.done:
                continuation = 5.0 * base_env.terminal_potential(step.next_state)
            else:
                continuation = min(
                    base_env._lookahead_score(step.next_state, float(next_action), depth - 1)
                    for next_action in base_env.local_action_candidates(step.next_state, action)
                )
            score += float(probability) * (float(step.cost) + penalty + continuation)
        return score

    def _build_smdp(
        self,
        blocks: Sequence[SymbolicBlock],
        absorption: Mapping[tuple[int, float], AbsorptionResult],
        frontier_transitions: Sequence[FrontierTransition],
    ) -> SMDPArrays:
        n_blocks = len(blocks)
        n_actions = len(self.env.actions)
        transitions = np.zeros((n_blocks, n_actions, n_blocks), dtype=float)
        costs = np.full((n_blocks, n_actions), np.inf, dtype=float)
        times = np.ones((n_blocks, n_actions), dtype=float)
        action_mask = np.ones((n_blocks, n_actions), dtype=bool)
        safety_idx = self.env.action_index(self.env.safety_action())
        frontier_support: dict[tuple[int, float], set[int]] = {}
        for transition in frontier_transitions:
            key = (transition.source_block, float(transition.action))
            frontier_support.setdefault(key, set()).add(transition.target_block)

        for block in blocks:
            if self.truncate_edges and block.is_truncated:
                action_mask[block.block_id, :] = False
                action_mask[block.block_id, safety_idx] = True
            for action_idx, action in enumerate(self.env.actions):
                if not action_mask[block.block_id, action_idx]:
                    transitions[block.block_id, action_idx, block.block_id] = 1.0
                    continue
                result = absorption[(block.block_id, float(action))]
                costs[block.block_id, action_idx] = result.expected_cost
                times[block.block_id, action_idx] = max(result.expected_time, 1e-9)
                allowed_targets = frontier_support.get((block.block_id, float(action)), set())
                for target_id, prob in result.hit_distribution.items():
                    if target_id != block.block_id and target_id not in allowed_targets:
                        raise FrontierSupportError(
                            "Absorption transition is not represented in the frontier graph used by the SMDP: "
                            f"B{block.block_id} --a={float(action):g}--> B{target_id}."
                        )
                    if 0 <= target_id < n_blocks:
                        transitions[block.block_id, action_idx, target_id] += prob
                total = transitions[block.block_id, action_idx].sum()
                if total <= 0:
                    transitions[block.block_id, action_idx, block.block_id] = 1.0
                else:
                    transitions[block.block_id, action_idx] /= total

        return SMDPArrays(
            transitions=transitions,
            costs=costs,
            times=times,
            action_mask=action_mask,
        )

    def _frontier_policy_iteration(self, smdp: SMDPArrays) -> tuple[np.ndarray, float, np.ndarray, int]:
        n_blocks = smdp.costs.shape[0]
        ratios = smdp.costs / np.maximum(smdp.times, 1e-9)
        ratios = np.where(smdp.action_mask, ratios, np.inf)
        policy = np.argmin(ratios, axis=1)
        eta, values = self._evaluate_policy(smdp, policy)

        for iteration in range(1, self.max_iter + 1):
            q = smdp.costs - eta * smdp.times + smdp.transitions @ values
            q = np.where(smdp.action_mask, q, np.inf)
            best_policy = np.argmin(q, axis=1)
            current_q = q[np.arange(n_blocks), policy]
            best_q = q[np.arange(n_blocks), best_policy]
            new_policy = np.where(current_q <= best_q + self.tolerance, policy, best_policy)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
            eta, values = self._evaluate_policy(smdp, policy)

        eta, values = self._evaluate_policy(smdp, policy)
        return policy, eta, values, iteration

    def _evaluate_policy(self, smdp: SMDPArrays, policy: np.ndarray) -> tuple[float, np.ndarray]:
        n = len(policy)
        idx = np.arange(n)
        p = smdp.transitions[idx, policy]
        c = smdp.costs[idx, policy]
        tau = np.maximum(smdp.times[idx, policy], 1e-9)

        a = p.T - np.eye(n)
        a[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        stationary = np.linalg.lstsq(a, b, rcond=None)[0]
        stationary = np.maximum(stationary, 0.0)
        total = stationary.sum()
        if total <= 0:
            stationary = np.ones(n) / n
        else:
            stationary /= total

        eta = float((stationary @ c) / max(float(stationary @ tau), 1e-12))

        poisson = np.eye(n) - p
        rhs = c - eta * tau
        poisson[0, :] = 0.0
        poisson[0, 0] = 1.0
        rhs[0] = 0.0
        values = np.linalg.lstsq(poisson, rhs, rcond=None)[0]
        values -= values[0]
        return eta, values
