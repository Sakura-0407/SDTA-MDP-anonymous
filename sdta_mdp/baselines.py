from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .environment import ContinuousEnvironment


PolicyFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BaselineResult:
    """统一保存一个算法或消融项的评估结果。"""

    benchmark: str
    method: str
    seed: int
    seconds: float
    iterations: int
    mean_cost: float | None
    success_rate: float | None
    violation_rate: float | None
    status: str
    model_calls: int = 0
    planning_horizon: int = 0
    execution_device: str = "cpu"
    blocks: int | None = None
    partition_seconds: float = 0.0
    preparation_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    transition_expectation: str = ""
    scenario_count: int = 0
    evaluation_protocol: str = ""


@dataclass(frozen=True)
class RolloutMetrics:
    """有限回合滚动评估指标。"""

    mean_cost: float
    success_rate: float
    violation_rate: float


def _expected_step_outcomes(
    env: ContinuousEnvironment,
    state: np.ndarray,
    action: float,
):
    expectation = getattr(env, "expected_step_outcomes", None)
    if callable(expectation):
        return tuple(expectation(state, action))
    return ((1.0, env.step(state, action)),)


def _sample_standard_noise_tape(
    env: ContinuousEnvironment,
    rng: np.random.Generator,
    *,
    scenarios: int,
    horizon: int,
) -> np.ndarray:
    sampler = getattr(env, "sample_standard_noise_tape", None)
    if callable(sampler):
        return np.asarray(sampler(rng, scenarios=scenarios, horizon=horizon), dtype=float)
    return np.zeros((scenarios, horizon, 2), dtype=float)


def _step_with_standard_noise(
    env: ContinuousEnvironment,
    state: np.ndarray,
    action: float,
    standard_noise: np.ndarray,
):
    stepper = getattr(env, "step_with_standard_noise", None)
    if callable(stepper):
        return stepper(state, action, standard_noise)
    return env.step(state, action)


class FineGridValueIteration:
    """对低维连续状态做均匀网格离散化的平均成本相对值迭代基线。"""

    def __init__(
        self,
        env: ContinuousEnvironment,
        *,
        bins_per_dim: int = 12,
        tolerance: float = 1e-6,
        max_iter: int = 1_000,
        discount: float = 0.98,
    ) -> None:
        self.env = env
        self.bins_per_dim = bins_per_dim
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.discount = discount
        self.axes = [
            np.linspace(float(low), float(high), bins_per_dim)
            for low, high in zip(env.bounds.low, env.bounds.high)
        ]
        self.shape = tuple(len(axis) for axis in self.axes)
        self.values = np.zeros(self.shape, dtype=float)
        self.policy_indices = np.zeros(self.shape, dtype=int)
        self.iterations = 0
        self.converged = False
        self.seconds = 0.0
        self.model_calls = 0

    def solve(self) -> "FineGridValueIteration":
        start = perf_counter()
        probabilities, costs, done, successors = self._build_transition_model()
        flat_state_count = int(np.prod(self.shape))
        for iteration in range(1, self.max_iter + 1):
            continuation = self.values.reshape(-1)[successors]
            q_values = np.sum(
                probabilities * (costs + self.discount * (1.0 - done) * continuation),
                axis=2,
            )
            flat_policy = np.argmin(q_values, axis=1)
            new_values = q_values[np.arange(flat_state_count), flat_policy].reshape(self.shape)
            new_policy = flat_policy.reshape(self.shape)
            if self.discount >= 1.0:
                reference = float(new_values.flat[0])
                new_values -= reference
            delta = float(np.max(np.abs(new_values - self.values)))
            self.values = new_values
            self.policy_indices = new_policy
            self.iterations = iteration
            if delta < self.tolerance:
                self.converged = True
                break
        self.seconds = perf_counter() - start
        return self

    def _build_transition_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows: list[list[tuple[tuple[float, object], ...]]] = []
        max_outcomes = 1
        for index in np.ndindex(self.shape):
            state = np.asarray(
                [self.axes[dim][idx] for dim, idx in enumerate(index)],
                dtype=float,
            )
            action_rows: list[tuple[tuple[float, object], ...]] = []
            for action in self.env.actions:
                outcomes = _expected_step_outcomes(self.env, state, float(action))
                action_rows.append(outcomes)
                max_outcomes = max(max_outcomes, len(outcomes))
                self.model_calls += len(outcomes)
            rows.append(action_rows)

        n_states = len(rows)
        n_actions = len(self.env.actions)
        probabilities = np.zeros((n_states, n_actions, max_outcomes), dtype=float)
        costs = np.zeros_like(probabilities)
        done = np.ones_like(probabilities)
        successors = np.zeros((n_states, n_actions, max_outcomes), dtype=int)
        for state_idx, action_rows in enumerate(rows):
            for action_idx, outcomes in enumerate(action_rows):
                for outcome_idx, (probability, step) in enumerate(outcomes):
                    probabilities[state_idx, action_idx, outcome_idx] = probability
                    costs[state_idx, action_idx, outcome_idx] = step.cost
                    done[state_idx, action_idx, outcome_idx] = float(step.done)
                    successors[state_idx, action_idx, outcome_idx] = np.ravel_multi_index(
                        self._nearest_indices(step.next_state),
                        self.shape,
                    )
        return probabilities, costs, done, successors

    def policy(self, state: np.ndarray) -> float:
        return float(self.env.actions[self.policy_indices[self._nearest_indices(state)]])

    def _nearest_indices(self, state: np.ndarray) -> tuple[int, ...]:
        clipped = self.env.bounds.clip(state)
        return tuple(int(np.argmin(np.abs(axis - clipped[dim]))) for dim, axis in enumerate(self.axes))


class TileCodingQLearning:
    """轻量 tile-coding Q-learning 基线，用于论文中的经典聚合对照。"""

    def __init__(
        self,
        env: ContinuousEnvironment,
        *,
        bins_per_dim: int = 8,
        episodes: int = 200,
        alpha: float = 0.25,
        gamma: float = 0.98,
        epsilon: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.bins_per_dim = bins_per_dim
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.shape = tuple([bins_per_dim] * len(env.state_names))
        self.q = np.zeros((*self.shape, len(env.actions)), dtype=float)
        self.seconds = 0.0
        self.model_calls = 0

    def train(self, max_steps: int = 120) -> "TileCodingQLearning":
        start = perf_counter()
        for _ in range(self.episodes):
            state = self.env.reset(self.rng)
            for _ in range(max_steps):
                tile = self._tile(state)
                if self.rng.random() < self.epsilon:
                    action_idx = int(self.rng.integers(len(self.env.actions)))
                else:
                    action_idx = int(np.argmin(self.q[tile]))
                step = self.env.step(state, float(self.env.actions[action_idx]))
                self.model_calls += 1
                target = step.cost
                if not step.done:
                    target += self.gamma * float(np.min(self.q[self._tile(step.next_state)]))
                self.q[(*tile, action_idx)] += self.alpha * (target - self.q[(*tile, action_idx)])
                state = step.next_state
                if step.done:
                    break
        self.seconds = perf_counter() - start
        return self

    def policy(self, state: np.ndarray) -> float:
        return float(self.env.actions[int(np.argmin(self.q[self._tile(state)]))])

    def _tile(self, state: np.ndarray) -> tuple[int, ...]:
        clipped = self.env.bounds.clip(state)
        scale = (clipped - self.env.bounds.low) / np.maximum(self.env.bounds.high - self.env.bounds.low, 1e-12)
        return tuple(np.minimum((scale * self.bins_per_dim).astype(int), self.bins_per_dim - 1).tolist())




def evaluate_policy(
    env: ContinuousEnvironment,
    policy: PolicyFn,
    *,
    episodes: int = 30,
    max_steps: int = 120,
    seed: int = 23,
) -> RolloutMetrics:
    rng = np.random.default_rng(seed)
    reset_action_diagnostics = getattr(env, "reset_evaluation_action_diagnostics", None)
    record_evaluation_action = getattr(env, "record_evaluation_action", None)
    if callable(reset_action_diagnostics):
        reset_action_diagnostics()
    costs: list[float] = []
    successes = 0
    violations = 0
    for _ in range(episodes):
        state = env.reset(rng)
        total = 0.0
        terminal = None
        violated = False
        for _ in range(max_steps):
            action = policy(state)
            step = env.step(state, action)
            if callable(record_evaluation_action):
                record_evaluation_action(step.info)
            total += step.cost
            state = step.next_state
            violated = violated or bool(step.info.get("constraint_violation", False))
            terminal = step.info.get("terminal")
            if step.done:
                break
        successes += int(terminal in {"goal", "stopped", "parked"} or env.distance_to_target(state) < 0.1)
        violations += int(violated or terminal == "collision")
        costs.append(total)
    return RolloutMetrics(
        mean_cost=float(np.mean(costs)),
        success_rate=successes / max(episodes, 1),
        violation_rate=violations / max(episodes, 1),
    )


def run_grid_vi_baseline(
    env: ContinuousEnvironment,
    *,
    episodes: int,
    bins_per_dim: int,
    seed: int,
    evaluation_env: ContinuousEnvironment | None = None,
) -> tuple[BaselineResult, PolicyFn]:
    total_start = perf_counter()
    preparation_start = perf_counter()
    solver = FineGridValueIteration(env, bins_per_dim=bins_per_dim).solve()
    preparation_seconds = perf_counter() - preparation_start
    evaluation_start = perf_counter()
    execution_env = evaluation_env or env
    metrics = evaluate_policy(execution_env, solver.policy, episodes=episodes, seed=seed)
    evaluation_seconds = perf_counter() - evaluation_start
    return (
        BaselineResult(
            execution_env.name,
            "FineGridVI",
            seed,
            perf_counter() - total_start,
            solver.iterations,
            metrics.mean_cost,
            metrics.success_rate,
            metrics.violation_rate,
            "ok" if solver.converged else "max-iter",
            model_calls=solver.model_calls,
            preparation_seconds=preparation_seconds,
            evaluation_seconds=evaluation_seconds,
            transition_expectation=(
                "discounted exact finite-action expectation"
                if env.transition_regime() == "stochastic" and not env.is_continuous_action()
                else "discounted 7-point Gauss-Hermite expectation"
                if env.transition_regime() == "stochastic"
                else "deterministic transition"
            ),
            evaluation_protocol="isolated execution environment" if evaluation_env is not None else "shared environment",
        ),
        solver.policy,
    )


def run_tile_q_baseline(
    env: ContinuousEnvironment,
    *,
    episodes: int,
    bins_per_dim: int,
    seed: int,
    evaluation_env: ContinuousEnvironment | None = None,
) -> tuple[BaselineResult, PolicyFn]:
    total_start = perf_counter()
    preparation_start = perf_counter()
    solver = TileCodingQLearning(env, bins_per_dim=max(4, bins_per_dim // 2), episodes=max(50, episodes * 20), seed=seed).train()
    preparation_seconds = perf_counter() - preparation_start
    evaluation_start = perf_counter()
    execution_env = evaluation_env or env
    metrics = evaluate_policy(execution_env, solver.policy, episodes=episodes, seed=seed)
    evaluation_seconds = perf_counter() - evaluation_start
    return (
        BaselineResult(
            execution_env.name,
            "TileQ",
            seed,
            perf_counter() - total_start,
            solver.episodes,
            metrics.mean_cost,
            metrics.success_rate,
            metrics.violation_rate,
            "ok",
            model_calls=solver.model_calls,
            preparation_seconds=preparation_seconds,
            evaluation_seconds=evaluation_seconds,
            transition_expectation="sampled Q-learning transitions",
            evaluation_protocol="isolated execution environment" if evaluation_env is not None else "shared environment",
        ),
        solver.policy,
    )






def run_mpc_baseline(
    env: ContinuousEnvironment,
    *,
    episodes: int,
    seed: int,
    horizon: int = 6,
    scenarios: int = 8,
    evaluation_env: ContinuousEnvironment | None = None,
) -> tuple[BaselineResult, PolicyFn]:
    """Scenario MPC with common random numbers across candidate actions."""

    start = perf_counter()
    model_calls = 0
    planning_rng = np.random.default_rng(seed + 104_729)
    scenario_count = scenarios if env.transition_regime() == "stochastic" else 1
    if scenario_count <= 0:
        raise ValueError("scenarios must be positive")

    def rollout_cost(state: np.ndarray, first_action: float, noise_tape: np.ndarray) -> float:
        nonlocal model_calls
        scenario_costs: list[float] = []
        for scenario_idx in range(scenario_count):
            current = env.bounds.clip(state)
            total = 0.0
            action = float(first_action)
            for step_idx in range(horizon):
                step = _step_with_standard_noise(
                    env,
                    current,
                    action,
                    noise_tape[scenario_idx, step_idx],
                )
                model_calls += 1
                total += step.cost
                if step.info.get("constraint_violation", False):
                    total += 100.0
                current = step.next_state
                if step.done:
                    break
                # Hold the candidate action over the short scenario horizon so
                # every counted model call belongs to the MPC baseline itself.
            scenario_costs.append(total + 5.0 * env.terminal_potential(current))
        return float(np.mean(scenario_costs))

    def policy(state: np.ndarray) -> float:
        candidates = env.local_action_candidates(state)
        noise_tape = _sample_standard_noise_tape(
            env,
            planning_rng,
            scenarios=scenario_count,
            horizon=horizon,
        )
        return float(
            min(
                candidates,
                key=lambda action: rollout_cost(state, float(action), noise_tape),
            )
        )

    evaluation_start = perf_counter()
    execution_env = evaluation_env or env
    metrics = evaluate_policy(execution_env, policy, episodes=episodes, seed=seed)
    evaluation_seconds = perf_counter() - evaluation_start
    result = BaselineResult(
        execution_env.name,
        "MPC",
        seed,
        perf_counter() - start,
        0,
        metrics.mean_cost,
        metrics.success_rate,
        metrics.violation_rate,
        "ok",
        model_calls=model_calls,
        planning_horizon=horizon,
        evaluation_seconds=evaluation_seconds,
        transition_expectation="scenario mean" if env.transition_regime() == "stochastic" else "deterministic rollout",
        scenario_count=scenario_count,
        evaluation_protocol="isolated execution environment" if evaluation_env is not None else "shared environment",
    )
    return result, policy
