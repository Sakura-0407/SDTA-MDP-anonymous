from __future__ import annotations
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from sdta_mdp.baselines import run_grid_vi_baseline, run_mpc_baseline, run_tile_q_baseline
from sdta_mdp.benchmarks import STOCHASTIC_MAIN_BENCHMARKS, make_benchmark
from sdta_mdp.evaluation import EvaluationConfig, _baseline_to_row, _evaluation_action_diagnostics, _configure_stochastic_noise, _run_sdta, _set_transition_seed

def run_stochastic_audit(config: EvaluationConfig, *, resume: bool=False, table_methods: dict[str, tuple[str, ...]] | None=None, slip_probability: float | None=None, continuous_noise_fraction: float | None=None, project_stochastic_paths: bool=False, gaussian_support_sigma: float=2.0, gaussian_support_points: int=3, support_aware_control: bool=False) -> list[dict[str, Any]]:
    method_map = table_methods or {b: ("SDTA-MDP", "MPC", "FineGridVI", "TileQ") for b in config.benchmarks}
    config.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.out_dir / 'results.csv'
    rows: list[dict[str, Any]] = read_csv(results_path) if resume else []
    if any((str(row.get('method', '')).startswith('SDTA') and (not str(row.get('model_call_accounting', '')).startswith('leaf-dynamics-v2:')) for row in rows)):
        raise ValueError('Legacy SDTA call counts cannot be resumed with complete accounting; use a fresh output directory')
    completed = {(str(row.get('benchmark', '')), str(row.get('method', '')), int(float(row.get('seed', -1)))) for row in rows if row.get('benchmark') and row.get('method') and (row.get('seed') not in {None, ''})}
    for benchmark in config.benchmarks:
        for seed in config.seeds:
            expected = _expected_methods_for(benchmark, method_map)
            missing = tuple((method for method in expected if (benchmark, method, seed) not in completed))
            if not missing:
                continue
            fresh_rows = _run_benchmark_seed(benchmark, seed, config, missing, slip_probability=slip_probability, continuous_noise_fraction=continuous_noise_fraction, project_stochastic_paths=project_stochastic_paths, gaussian_support_sigma=gaussian_support_sigma, gaussian_support_points=gaussian_support_points, support_aware_control=support_aware_control)
            for row in fresh_rows:
                key = (str(row['benchmark']), str(row['method']), int(row['seed']))
                rows = [existing for existing in rows if (str(existing.get('benchmark', '')), str(existing.get('method', '')), int(float(existing.get('seed', -1)))) != key]
                rows.append(row)
                completed.add(key)
            write_csv(results_path, rows)
    return rows

def _expected_methods_for(benchmark: str, table_methods: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return table_methods[benchmark]

def _run_benchmark_seed(benchmark: str, seed: int, config: EvaluationConfig, expected_methods: tuple[str, ...], *, slip_probability: float | None=None, continuous_noise_fraction: float | None=None, project_stochastic_paths: bool=False, gaussian_support_sigma: float=2.0, gaussian_support_points: int=3, support_aware_control: bool=False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if 'SDTA-MDP' in expected_methods:
        rows.append(asdict(_run_sdta(env_name=benchmark, seed=seed, config=config, method='SDTA-MDP', truncate_edges=True, absorption_samples=config.absorption_samples, slip_probability=slip_probability, continuous_noise_fraction=continuous_noise_fraction, project_stochastic_paths=project_stochastic_paths, gaussian_support_sigma=gaussian_support_sigma, gaussian_support_points=gaussian_support_points, support_aware_control=support_aware_control)))
    if 'SDTA-NoLookahead' in expected_methods:
        rows.append(asdict(_run_sdta(env_name=benchmark, seed=seed, config=config, method='SDTA-NoLookahead', truncate_edges=True, absorption_samples=config.absorption_samples, disable_lookahead=True, slip_probability=slip_probability, continuous_noise_fraction=continuous_noise_fraction, project_stochastic_paths=project_stochastic_paths, gaussian_support_sigma=gaussian_support_sigma, gaussian_support_points=gaussian_support_points, support_aware_control=support_aware_control)))
    for method in ('FineGridVI', 'TileQ', 'MPC'):
        if method not in expected_methods:
            continue
        env = make_benchmark(benchmark)
        execution_env = make_benchmark(benchmark)
        _configure_stochastic_noise(env, slip_probability=slip_probability, continuous_noise_fraction=continuous_noise_fraction)
        _configure_stochastic_noise(execution_env, slip_probability=slip_probability, continuous_noise_fraction=continuous_noise_fraction)
        _set_transition_seed(env, seed + 1000003)
        _set_transition_seed(execution_env, seed)
        if method == 'FineGridVI':
            result, _ = run_grid_vi_baseline(env, episodes=config.episodes, bins_per_dim=config.grid_bins, seed=seed, evaluation_env=execution_env)
        elif method == 'TileQ':
            result, _ = run_tile_q_baseline(env, episodes=config.episodes, bins_per_dim=config.grid_bins, seed=seed, evaluation_env=execution_env)
        else:
            result, _ = run_mpc_baseline(env, episodes=config.episodes, seed=seed, horizon=config.mpc_horizon, scenarios=config.mpc_scenarios, evaluation_env=execution_env)
        row = _baseline_to_row(result, grid_bins=config.grid_bins)
        row.update(_evaluation_action_diagnostics(execution_env))
        row['noise_scale'] = execution_env.transition_noise_scale()
        rows.append(row)
    return rows

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))
