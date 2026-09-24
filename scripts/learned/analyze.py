import csv
import json

import run as r


def analyze():
    manifest = json.loads((r.OUT / 'manifest.json').read_text())
    assert r.source_hashes() == manifest['sources'], 'Frozen source changed'
    rows, metrics, replayed = [], ('cost', 'success', 'violation', 'strict_success', 'reach_avoid'), 0
    for job in manifest['jobs']:
        folder = r.OUT / 'jobs' / job['id']
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'complete' and status['step'] == manifest['steps']
        assert status['gradient_updates'] == manifest['steps'] - manifest['warmup']
        source = r.ROOT / job['input_source']
        inputs = r.OUT / 'inputs' / f"{job['task']}_{job['seed']}.npz"
        assert r.sha(source) == r.sha(inputs) == job['input_sha']
        with r.np.load(inputs, allow_pickle=False) as data:
            initial, noise = data['initial'], data['tape']
        for step in manifest['checkpoints']:
            row = json.loads((folder / f'result_{step}.json').read_text())
            assert row['training_steps'] == row['training_interactions'] == step
            assert row['input_sha'] == job['input_sha'] and row['seed'] == job['seed']
            with (folder / f'episodes_{step}.csv').open(newline='', encoding='utf-8') as stream:
                episodes = list(csv.DictReader(stream))
            assert len(episodes) == 20 and {int(e['episode']) for e in episodes} == set(range(20))
            for key in metrics:
                r.np.testing.assert_allclose(row[key], r.np.mean([float(e[key]) for e in episodes]), atol=1e-12)
            if step != manifest['final_checkpoint']:
                continue
            env = r.environment(job['task'], job['seed'])
            with r.np.load(folder / f'trace_{step}.npz', allow_pickle=False) as data:
                trace = data['trace']
            assert r.np.isfinite(trace).all()
            for episode in episodes:
                ep = int(episode['episode'])
                path = trace[trace[:, 0] == ep]
                assert len(path) == int(episode['steps']) and 1 <= len(path) <= 120
                state, cost, violation, terminal = initial[ep].copy(), 0., False, None
                for t, sample in enumerate(path):
                    assert sample[1] == t
                    r.np.testing.assert_array_equal(state, sample[2:4])
                    r.np.testing.assert_array_equal(noise[ep, t], sample[6:8])
                    result = env.step_with_standard_noise(state, sample[4], noise[ep, t])
                    r.np.testing.assert_array_equal(result.next_state, sample[8:10])
                    assert result.info['executed_action'] == sample[5]
                    assert result.cost == sample[10] and result.done == bool(sample[11])
                    assert not result.done or t == len(path) - 1
                    cost += result.cost
                    terminal = result.info.get('terminal')
                    violation |= bool(result.info.get('constraint_violation', False)) or terminal == 'collision'
                    state = result.next_state
                    replayed += 1
                assert result.done or len(path) == 120
                strict = terminal in {'goal', 'stopped', 'parked'}
                success = strict or env.distance_to_target(state) < .1
                assert float(episode['success']) == float(success)
                assert float(episode['violation']) == float(violation)
                assert float(episode['strict_success']) == float(strict)
                assert float(episode['reach_avoid']) == float(strict and not violation)
                r.np.testing.assert_allclose(float(episode['cost']), cost, rtol=0, atol=1e-12)
            rows.append(row)
    assert len(rows) == 80
    assert len({(x['task'], x['method'], x['seed']) for x in rows}) == 80
    r.write_csv(r.OUT / 'final_results.csv', rows)
    summary = []
    rng = r.np.random.default_rng(20260923)
    for task in r.TASKS:
        for method in ('SMoSE', 'SAC'):
            group = sorted([x for x in rows if x['task'] == task and x['method'] == method], key=lambda x: x['seed'])
            assert [x['seed'] for x in group] == list(r.SEEDS)
            for metric in (*metrics, 'training_interactions', 'evaluation_interactions', 'elapsed_seconds'):
                values = r.np.asarray([x[metric] for x in group], dtype=float)
                boot = values[rng.integers(0, 10, size=(10000, 10))].mean(axis=1)
                low, high = r.np.quantile(boot, [.025, .975])
                summary.append(dict(task=task, method=method, metric=metric, seeds=10,
                    mean=float(values.mean()), std=float(values.std(ddof=1)),
                    ci95_low=float(low), ci95_high=float(high)))
    r.write_csv(r.OUT / 'summary.csv', summary)
    changed = [path for path, digest in manifest['protected_files'].items()
               if not (r.ROOT / path).exists() or r.sha(r.ROOT / path) != digest]
    r.dump(r.OUT / 'audit.json', dict(status='passed', jobs=len(rows), evaluation_rows=240,
        final_evaluation_episodes=1600, replayed_transitions=replayed, source_hashes_match=True,
        evaluation_inputs_match=True, protected_files_changed=changed,
        note='Protected-file changes, if any, are reported; this runner never writes manuscript files.'))


if __name__ == '__main__':
    try:
        analyze()
    except Exception as exc:
        r.dump(r.OUT / 'audit.json', dict(status='failed', error=repr(exc)))
        raise
