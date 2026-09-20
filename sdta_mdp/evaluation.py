from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .baselines import BaselineResult, evaluate_policy
from .benchmarks import make_benchmark
from .solver import DTPTASolver
from .model_calls import ModelCallCounter
from .visualization import write_solver_visualizations


@dataclass(frozen=True)
class EvaluationConfig:
    """论文实验管线参数。"""
    benchmarks: tuple[str, ...] = ('braking_car',)
    seeds: tuple[int, ...] = (0,)
    episodes: int = 20
    absorption_samples: int = 32
    absorption_horizon: int = 80
    frontier_mode: str = 'sampled'
    grid_bins: int = 8
    mpc_horizon: int = 6
    mpc_scenarios: int = 8
    out_dir: Path = Path('runs/main')


@dataclass(frozen=True)
class SDTARow:
    """SDTA-MDP 及其消融项的一行实验结果。"""

    benchmark: str
    method: str
    seed: int
    seconds: float
    preparation_seconds: float
    evaluation_seconds: float
    iterations: int
    mean_cost: float
    success_rate: float
    violation_rate: float
    blocks: int
    core_blocks: int
    truncated_blocks: int
    frontier_transitions: int
    absorption_samples: int
    status: str
    model_calls: int
    partition_seconds: float
    absorption_seconds: float
    frontier_solve_seconds: float
    frontier_mode: str = "auto"
    frontier_backend: str = "unknown"
    exact_action_projection: bool = False
    exact_projection_available: bool = False
    exact_projection_mean_width: float | None = None
    exact_projection_feasible_rate: float | None = None
    exact_projection_activation_rate: float | None = None
    exact_projection_clipping_rate: float | None = None
    action_regime: str = "finite-action"
    control_route: str = "finite-core"
    execution_device: str = "cpu"
    scale_size: int | None = None
    grid_state_count: int | None = None
    block_compression_ratio: float | None = None
    scalability_family: str = ""
    transition_regime: str = "deterministic"
    noise_scale: float | None = None
    noise_model: str = ""
    action_sample_count: int = 0
    action_perturbation_count: int = 0
    action_perturbation_rate: float | None = None
    mean_abs_action_deviation: float | None = None
    mean_intended_action: float | None = None
    mean_executed_action: float | None = None
    frontier_hit_entropy: float | None = None
    transition_expectation: str = ""
    scenario_count: int = 0
    evaluation_protocol: str = ""
    topology_noise_projection: str = ""
    topology_noise_interval_probability: float | None = None
    topology_noise_support_points: int | None = None
    support_aware_control: bool = False
    preparation_model_calls: int = 0
    absorption_model_calls: int = 0
    other_preparation_model_calls: int = 0
    online_model_calls: int = 0
    policy_decisions: int = 0
    online_model_calls_per_decision: float = 0.0
    model_call_accounting: str = ""












def _run_sdta(
    *,
    env_name: str,
    seed: int,
    config: EvaluationConfig,
    method: str,
    truncate_edges: bool,
    absorption_samples: int,
    fixed_actions: bool = False,
    disable_lookahead: bool = False,
    absorption_horizon: int | None = None,
    exact_actions: bool = False,
    slip_probability: float | None = None,
    continuous_noise_fraction: float | None = None,
    project_stochastic_paths: bool = False,
    gaussian_support_sigma: float = 2.0,
    gaussian_support_points: int = 3,
    support_aware_control: bool = False,
) -> SDTARow:
    env = make_benchmark(env_name)
    _configure_stochastic_noise(
        env,
        slip_probability=slip_probability,
        continuous_noise_fraction=continuous_noise_fraction,
        project_stochastic_paths=project_stochastic_paths,
        gaussian_support_sigma=gaussian_support_sigma,
        gaussian_support_points=gaussian_support_points,
    )
    _set_transition_seed(env, seed)
    execution_env = make_benchmark(env_name) if env.transition_regime() == "stochastic" else env
    if execution_env is not env:
        _configure_stochastic_noise(
            execution_env,
            slip_probability=slip_probability,
            continuous_noise_fraction=continuous_noise_fraction,
            project_stochastic_paths=project_stochastic_paths,
            gaussian_support_sigma=gaussian_support_sigma,
            gaussian_support_points=gaussian_support_points,
        )
        _set_transition_seed(execution_env, seed)
    if fixed_actions and env.is_continuous_action():
        low, high = env.action_bounds()
        env.actions = np.linspace(float(low[0]), float(high[0]), 5)
        env.action_candidates = lambda fallback_action=None: env.actions  # type: ignore[method-assign]
        env.state_critical_actions = lambda state: ()  # type: ignore[method-assign]
    if disable_lookahead:
        env.local_policy_action = lambda state, fallback_action=None: float(fallback_action if fallback_action is not None else env.safety_action())  # type: ignore[method-assign]
    projection_stats: dict[str, Any] = {"calls": 0, "feasible": 0, "activated": 0, "clipped": 0, "widths": []}
    if exact_actions:
        exact_policy_action = env.exact_policy_action

        def instrumented_exact_policy_action(state, fallback_action=None):
            interval = env.exact_safe_action_interval(state)
            projection_stats["calls"] += 1
            if interval is not None and interval.feasible:
                projection_stats["feasible"] += 1
                projection_stats["activated"] += 1
                projection_stats["widths"].append(max(0.0, float(interval.high) - float(interval.low)))
                if fallback_action is not None and not (interval.low - 1e-9 <= float(fallback_action) <= interval.high + 1e-9):
                    projection_stats["clipped"] += 1
            return exact_policy_action(state, fallback_action)

        env.local_policy_action = instrumented_exact_policy_action  # type: ignore[method-assign]
    start = perf_counter()
    solver = DTPTASolver(
        env,
        absorption_samples=absorption_samples,
        absorption_horizon=absorption_horizon or config.absorption_horizon,
        frontier_mode=config.frontier_mode,
        truncate_edges=truncate_edges,
        support_aware_control=support_aware_control,
        seed=seed,
    )
    with ModelCallCounter(env) as counter:
        with counter.measure("preparation"):
            result = solver.solve()
        preparation_seconds = perf_counter() - start
        evaluation_start = perf_counter()
        policy = solver.make_policy(result)
        policy_decisions = 0

        def counted_policy(state):
            nonlocal policy_decisions
            policy_decisions += 1
            with counter.measure("online"):
                return policy(state)

        metrics = evaluate_policy(execution_env, counted_policy, episodes=config.episodes, seed=seed)
    action_diagnostics = _evaluation_action_diagnostics(execution_env)
    evaluation_seconds = perf_counter() - evaluation_start
    seconds = perf_counter() - start
    if method == "SDTA-MDP":
        write_solver_visualizations(config.out_dir / "figures", env, result, solver.make_policy(result), seed=seed)
    scale_metadata = _scalability_metadata(env, grid_bins=config.grid_bins, blocks=len(result.blocks))
    return SDTARow(
        benchmark=execution_env.name,
        method=method,
        seed=seed,
        seconds=seconds,
        preparation_seconds=preparation_seconds,
        evaluation_seconds=evaluation_seconds,
        iterations=result.iterations,
        mean_cost=metrics.mean_cost,
        success_rate=metrics.success_rate,
        violation_rate=metrics.violation_rate,
        blocks=len(result.blocks),
        core_blocks=sum(block.role == "core" for block in result.blocks),
        truncated_blocks=sum(block.is_truncated for block in result.blocks),
        frontier_transitions=len(result.partition_report.frontier_transitions),
        absorption_samples=absorption_samples,
        status="ok",
        model_calls=counter.total,
        preparation_model_calls=counter.calls["preparation"],
        absorption_model_calls=result.model_calls,
        other_preparation_model_calls=counter.calls["preparation"] - result.model_calls,
        online_model_calls=counter.calls["online"],
        policy_decisions=policy_decisions,
        online_model_calls_per_decision=counter.calls["online"] / max(policy_decisions, 1),
        model_call_accounting="leaf-dynamics-v2: preparation + online; excludes execution and visualization",
        partition_seconds=result.partition_seconds,
        absorption_seconds=result.absorption_seconds,
        frontier_solve_seconds=result.frontier_solve_seconds,
        frontier_mode=result.partition_report.frontier_mode,
        frontier_backend=result.partition_report.frontier_backend,
        exact_action_projection=exact_actions,
        exact_projection_available=env.supports_exact_action_projection(),
        exact_projection_mean_width=_projection_mean_width(projection_stats),
        exact_projection_feasible_rate=_projection_rate(projection_stats, "feasible"),
        exact_projection_activation_rate=_projection_rate(projection_stats, "activated"),
        exact_projection_clipping_rate=_projection_rate(projection_stats, "clipped"),
        action_regime=_action_regime(env.name),
        control_route=_control_route(method, env.name),
        execution_device="cpu",
        scale_size=scale_metadata["scale_size"],
        grid_state_count=scale_metadata["grid_state_count"],
        block_compression_ratio=scale_metadata["block_compression_ratio"],
        scalability_family=scale_metadata["scalability_family"],
        transition_regime=execution_env.transition_regime(),
        noise_scale=execution_env.transition_noise_scale(),
        noise_model=str(action_diagnostics["noise_model"]),
        topology_noise_projection=str(
            getattr(env, "topology_noise_projection", lambda: "none")()
        ),
        topology_noise_interval_probability=getattr(
            env,
            "topology_noise_interval_probability",
            lambda: None,
        )(),
        topology_noise_support_points=getattr(
            env,
            "topology_noise_support_points",
            lambda: None,
        )(),
        support_aware_control=bool(support_aware_control),
        action_sample_count=int(action_diagnostics["action_sample_count"]),
        action_perturbation_count=int(action_diagnostics["action_perturbation_count"]),
        action_perturbation_rate=_to_optional_float(action_diagnostics["action_perturbation_rate"]),
        mean_abs_action_deviation=_to_optional_float(action_diagnostics["mean_abs_action_deviation"]),
        mean_intended_action=_to_optional_float(action_diagnostics["mean_intended_action"]),
        mean_executed_action=_to_optional_float(action_diagnostics["mean_executed_action"]),
        frontier_hit_entropy=_mean_frontier_hit_entropy(result.absorption_results),
        transition_expectation="Monte Carlo absorption estimates",
        scenario_count=absorption_samples,
        evaluation_protocol=(
            "isolated execution environment with common per-seed disturbance stream"
            if execution_env is not env
            else "shared environment"
        ),
    )


def _evaluation_action_diagnostics(env: Any) -> dict[str, Any]:
    getter = getattr(env, "evaluation_action_diagnostics", None)
    if callable(getter):
        return dict(getter())
    return {
        "noise_model": "",
        "action_sample_count": 0,
        "action_perturbation_count": 0,
        "action_perturbation_rate": None,
        "mean_abs_action_deviation": None,
        "mean_intended_action": None,
        "mean_executed_action": None,
    }


def _to_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _baseline_to_row(result: BaselineResult, *, grid_bins: int | None = None) -> dict[str, Any]:
    row = asdict(result)
    benchmark = str(row.get("benchmark", ""))
    scale_metadata = _scalability_metadata(benchmark, grid_bins=grid_bins)
    row.update(
        {
            "blocks": row.get("blocks") if row.get("blocks") is not None else "",
            "core_blocks": "",
            "truncated_blocks": "",
            "frontier_transitions": "",
            "absorption_samples": "",
            "partition_seconds": row.get("partition_seconds", ""),
            "absorption_seconds": "",
            "frontier_solve_seconds": "",
            "frontier_mode": "",
            "frontier_backend": "",
            "exact_action_projection": "",
            "exact_projection_available": _exact_projection_available(benchmark),
            "exact_projection_mean_width": "",
            "exact_projection_feasible_rate": "",
            "exact_projection_activation_rate": "",
            "exact_projection_clipping_rate": "",
            "action_regime": _action_regime(benchmark),
            "control_route": _control_route(str(row.get("method", "")), benchmark),
            "scale_size": scale_metadata["scale_size"],
            "grid_state_count": scale_metadata["grid_state_count"],
            "block_compression_ratio": scale_metadata["block_compression_ratio"],
            "scalability_family": scale_metadata["scalability_family"],
            "transition_regime": _transition_regime(benchmark),
            "noise_scale": _noise_scale(benchmark),
            "frontier_hit_entropy": "",
        }
    )
    return row


def _set_transition_seed(env: Any, seed: int) -> None:
    setter = getattr(env, "set_transition_seed", None)
    if callable(setter):
        setter(int(seed))


def _configure_stochastic_noise(
    env: Any,
    *,
    slip_probability: float | None = None,
    continuous_noise_fraction: float | None = None,
    project_stochastic_paths: bool | None = None,
    gaussian_support_sigma: float | None = None,
    gaussian_support_points: int | None = None,
) -> None:
    """Apply an explicit stochastic-action strength without rebuilding a benchmark."""
    if getattr(env, "transition_regime", lambda: "deterministic")() != "stochastic":
        return
    if slip_probability is not None:
        if not 0.0 <= slip_probability <= 1.0:
            raise ValueError("slip_probability must lie in [0, 1]")
        env.slip_probability = float(slip_probability)
    if continuous_noise_fraction is not None:
        if continuous_noise_fraction < 0.0:
            raise ValueError("continuous_noise_fraction must be non-negative")
        env.continuous_noise_fraction = float(continuous_noise_fraction)
    if project_stochastic_paths is not None and hasattr(env, "project_stochastic_paths"):
        env.project_stochastic_paths = bool(project_stochastic_paths)
    if gaussian_support_sigma is not None and hasattr(env, "gaussian_support_sigma"):
        if gaussian_support_sigma <= 0.0:
            raise ValueError("gaussian_support_sigma must be positive")
        env.gaussian_support_sigma = float(gaussian_support_sigma)
    if gaussian_support_points is not None and hasattr(env, "gaussian_support_points"):
        if gaussian_support_points < 2:
            raise ValueError("gaussian_support_points must be at least two")
        env.gaussian_support_points = int(gaussian_support_points)


def _transition_regime(benchmark: str) -> str:
    if not benchmark:
        return "deterministic"
    try:
        return make_benchmark(benchmark).transition_regime()
    except ValueError:
        return "deterministic"


def _noise_scale(benchmark: str) -> float | None:
    if not benchmark:
        return None
    try:
        value = make_benchmark(benchmark).transition_noise_scale()
    except ValueError:
        return None
    return float(value) if value else None


def _mean_frontier_hit_entropy(absorption: Mapping[tuple[int, float], Any]) -> float | None:
    entropies: list[float] = []
    for result in absorption.values():
        probs = [float(prob) for prob in result.hit_distribution.values() if float(prob) > 0.0]
        if not probs:
            continue
        entropies.append(float(-sum(prob * np.log2(prob) for prob in probs)))
    if not entropies:
        return None
    return float(mean(entropies))


def _scalability_metadata(
    env_or_benchmark: Any,
    *,
    grid_bins: int | None,
    blocks: int | None = None,
) -> dict[str, Any]:
    try:
        env = make_benchmark(env_or_benchmark) if isinstance(env_or_benchmark, str) else env_or_benchmark
    except ValueError:
        return {"scale_size": None, "grid_state_count": None, "block_compression_ratio": None, "scalability_family": ""}
    family = str(getattr(env, "scalability_family", ""))
    scale_size = getattr(env, "scale_size", None)
    grid_state_count = None
    if grid_bins is not None:
        grid_state_count = int(grid_bins) ** len(env.state_names)
    ratio = None
    if family and grid_state_count is not None and blocks is not None and blocks > 0:
        ratio = float(grid_state_count) / float(blocks)
    return {
        "scale_size": int(scale_size) if scale_size is not None else None,
        "grid_state_count": grid_state_count if family else None,
        "block_compression_ratio": ratio,
        "scalability_family": family,
    }


def _projection_mean_width(stats: dict[str, Any]) -> float | None:
    widths = stats.get("widths") or []
    if not widths:
        return None
    return float(mean(float(width) for width in widths))


def _projection_rate(stats: dict[str, Any], key: str) -> float | None:
    calls = int(stats.get("calls") or 0)
    if calls <= 0:
        return None
    return float(stats.get(key, 0)) / calls


def _action_regime(benchmark: str) -> str:
    name = str(benchmark)
    try:
        return "continuous-action" if make_benchmark(name).is_continuous_action() else "finite-action"
    except ValueError:
        pass
    return "continuous-action" if name.startswith("continuous_") or name.startswith("large_continuous_") or name.startswith("stochastic_continuous_") or name.endswith("_continuous") or name == "pendulum_swing_up" else "finite-action"


def _exact_projection_available(benchmark: str) -> bool:
    if not benchmark:
        return False
    try:
        return make_benchmark(benchmark).supports_exact_action_projection()
    except ValueError:
        return False


def _control_route(method: str, benchmark: str) -> str:
    if method == "SDTA-MDP-ExactAction":
        return "exact-smt-projection"
    if method == "SDTA-MDP":
        return "adaptive-candidates" if _action_regime(benchmark) == "continuous-action" else "finite-core"
    if method == "SDTA-FixedActions":
        return "fixed-candidates"
    if method == "SDTA-NoLookahead":
        return "adaptive-candidates"
    if method.startswith("SDTA"):
        return "adaptive-candidates" if _action_regime(benchmark) == "continuous-action" else "finite-core"
    return "baseline"












def _paired_sign_flip_pvalue(differences: list[float], *, samples: int = 10_000, seed: int = 2026) -> float:
    clean = [float(value) for value in differences if np.isfinite(value)]
    if not clean:
        return 1.0
    observed = abs(mean(clean))
    if observed <= 1e-12:
        return 1.0
    if len(clean) <= 16:
        total = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=len(clean)):
            statistic = abs(mean(sign * diff for sign, diff in zip(signs, clean)))
            extreme += int(statistic >= observed - 1e-12)
            total += 1
        return extreme / total

    rng = np.random.default_rng(seed)
    diffs = np.asarray(clean, dtype=float)
    signs = rng.choice((-1.0, 1.0), size=(samples, len(clean)))
    statistics = np.abs(np.mean(signs * diffs, axis=1))
    return float((np.count_nonzero(statistics >= observed - 1e-12) + 1) / (samples + 1))




























def _bootstrap_ci(values: list[float], *, samples: int = 1_000, seed: int = 2026) -> tuple[float, float]:
    """使用固定随机种子生成可复现的均值 bootstrap 区间。"""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
