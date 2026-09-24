"""Evaluate final SAC/SMoSE actors under the original manuscript protocol."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import run as r
from sdta_mdp.baselines import evaluate_policy


def main_inputs(task, seed):
    env, rng = r.environment(task, seed), r.np.random.default_rng(seed)
    initial, tapes = [], []
    for _ in range(20):
        initial.append(env.reset(rng))
        tape = r.np.zeros((120, 2))
        tape[:, 0] = env._transition_rng.normal(size=120)
        tapes.append(tape)
    return r.np.asarray(initial), r.np.asarray(tapes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=r.ROOT/'reference/learned/policies')
    parser.add_argument('--out-dir', type=Path, default=r.ROOT/'runs/learned_evaluation')
    args = parser.parse_args()
    jobs = json.loads((r.ROOT/'reference/learned/jobs.json').read_text())
    reference = r.ROOT/'reference/learned/results.csv'
    with reference.open(newline='', encoding='utf-8') as f:
        recorded = {(x['task'], x['method'], int(x['seed'])):x for x in csv.DictReader(f)}
    is_recorded = args.checkpoint_dir.resolve() == (r.ROOT/'reference/learned/policies').resolve()
    rows = []
    for job in jobs:
        task, method, seed = job['task'], job['method'], job['seed']
        folder = args.checkpoint_dir/job['id']
        compact = folder/'actor.pt'
        if compact.exists():
            state = r.torch.load(compact, weights_only=True, map_location='cpu')
            actor_state = state['actor']
        else:
            state = r.torch.load(folder/'resume.pt', weights_only=False, map_location='cpu')
            actor_state = state['modules']['actor']
            assert state['env']['calls'] == 100000
        assert state['step'] == 100000
        _, cfg = r.configuration(task, method, job['training_seed'])
        env = r.make_external_env(task, job['training_seed'])
        actor = (r.MLPActor if method == 'SAC' else r.Actor)(cfg, env)
        actor.load_state_dict(actor_state)
        actor.eval()
        initial, tape = main_inputs(task, seed)
        with r.np.load(r.ROOT/f'reference/learned/main_inputs/{task}_{seed}.npz') as archived:
            r.np.testing.assert_array_equal(initial, archived['initial'])
            r.np.testing.assert_array_equal(tape, archived['tape'])
        metrics = r.evaluate(cfg, env, SimpleNamespace(actor=actor), initial, tape,
                             args.out_dir/'jobs'/job['id'], 100000)
        with r.torch.no_grad():
            check = evaluate_policy(r.environment(task, seed),
                lambda s: float(actor.get_action(r.torch.as_tensor(s[None], dtype=r.torch.float32),
                                                deterministic=True)[0].item()), episodes=20, seed=seed)
        r.np.testing.assert_allclose([metrics['cost'], metrics['success'], metrics['violation']],
                                    [check.mean_cost, check.success_rate, check.violation_rate], rtol=0, atol=1e-12)
        if is_recorded:
            expected = recorded[task, method, seed]
            assert state['source_checkpoint_sha256'] == expected['checkpoint_sha']
            keys = ('cost', 'success', 'strict_success', 'violation', 'reach_avoid', 'steps')
            r.np.testing.assert_allclose([metrics[k] for k in keys],
                                        [float(expected[k]) for k in keys], rtol=0, atol=1e-10)
        rows.append(dict(task=task, method=method, seed=seed, **metrics,
                         training_interactions=100000, online_model_predictions=0,
                         evaluation_interactions=int(round(metrics['steps']*20))))
        print(f'{len(rows)}/80 {job["id"]}: verified', flush=True)
    assert len(rows) == len({(x['task'],x['method'],x['seed']) for x in rows}) == 80
    r.write_csv(args.out_dir/'results.csv', rows)
    r.dump(args.out_dir/'audit.json', dict(status='passed', jobs=80, episodes=1600,
        original_evaluator_checks=80, recorded_metrics_matched=is_recorded, training_performed=False))


if __name__ == '__main__':
    main()
