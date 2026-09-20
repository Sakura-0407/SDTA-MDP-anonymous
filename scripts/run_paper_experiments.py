"""Run only the seven-task main study or the four-task lookahead ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdta_mdp.benchmarks import STOCHASTIC_MAIN_BENCHMARKS
from sdta_mdp.evaluation import EvaluationConfig
from sdta_mdp.paper_experiments import run_stochastic_audit

CONTINUOUS = tuple(b for b in STOCHASTIC_MAIN_BENCHMARKS if b in {
    'stochastic_continuous_braking_car', 'stochastic_pendulum_swing_up',
    'stochastic_continuous_point_mass', 'stochastic_continuous_double_integrator_parking'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['main', 'lookahead'], default='main')
    parser.add_argument('--benchmarks', nargs='+', choices=STOCHASTIC_MAIN_BENCHMARKS)
    parser.add_argument('--methods', nargs='+', choices=['SDTA-MDP', 'MPC', 'FineGridVI', 'TileQ', 'SDTA-NoLookahead'])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--absorption-samples', type=int, default=32)
    parser.add_argument('--absorption-horizon', type=int, default=80)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    benchmarks = tuple(args.benchmarks or (STOCHASTIC_MAIN_BENCHMARKS if args.suite == 'main' else CONTINUOUS))
    methods = tuple(args.methods or (('SDTA-MDP','MPC','FineGridVI','TileQ') if args.suite == 'main' else ('SDTA-MDP','SDTA-NoLookahead')))
    if args.suite == 'lookahead' and any(b not in CONTINUOUS for b in benchmarks):
        parser.error('The manuscript lookahead ablation covers only four continuous-action tasks.')
    if args.suite == 'main' and 'SDTA-NoLookahead' in methods:
        parser.error('Use --suite lookahead for the ablation.')
    if args.suite == 'lookahead' and any(m not in {'SDTA-MDP','SDTA-NoLookahead'} for m in methods):
        parser.error('The lookahead suite compares SDTA-MDP and SDTA-NoLookahead only.')
    if min(args.episodes,args.absorption_samples,args.absorption_horizon) < 1 or not args.seeds:
        parser.error('Episode/sample/horizon counts must be positive.')
    out = args.out_dir or ROOT / 'runs' / args.suite
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/'configs/paper.json').read_text())
    protocol.update(benchmarks=benchmarks, methods=methods, seeds=args.seeds,
                    episodes=args.episodes, absorption_samples=args.absorption_samples,
                    absorption_horizon=args.absorption_horizon, suite=args.suite)
    protocol['source_hashes'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'sdta_mdp').glob('*.py'))}
    config_path = out/'protocol.json'
    # Normalize tuples to JSON arrays before comparing resumed protocols.
    protocol = json.loads(json.dumps(protocol))
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != protocol:
            raise ValueError('Output already exists or protocol differs; use a fresh directory.')
    elif (out/'results.csv').exists():
        raise ValueError('Unmanifested results cannot be resumed.')
    config_path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    config = EvaluationConfig(benchmarks=benchmarks, seeds=tuple(args.seeds),
        episodes=args.episodes, absorption_samples=args.absorption_samples,
        absorption_horizon=args.absorption_horizon, frontier_mode='sampled',
        grid_bins=8, mpc_horizon=6, mpc_scenarios=8, out_dir=out)
    rows = run_stochastic_audit(config, resume=args.resume,
        table_methods={b:methods for b in benchmarks}, slip_probability=0.10,
        continuous_noise_fraction=0.05, project_stochastic_paths=True,
        gaussian_support_sigma=2.0, gaussian_support_points=3, support_aware_control=True)
    expected = len(benchmarks)*len(args.seeds)*len(methods)
    if len(rows) != expected:
        raise RuntimeError(f'Expected {expected} records, received {len(rows)}.')
    print(f'Completed {len(rows)} records in {out}')


if __name__ == '__main__':
    main()
