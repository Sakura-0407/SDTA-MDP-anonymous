from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runs/learned'
PRIOR = ROOT / 'reference/learned/training_inputs'
sys.path.insert(0, str(ROOT / 'external/SMoSE/src'))
import numpy as np
import torch
import yaml
from sac import setup_sac, train_sac, Actor, MLPActor
from adapters import environment, make_external_env, dump, write_csv, sha

torch.set_num_threads(1)
TASKS = ('stochastic_continuous_braking_car', 'stochastic_pendulum_swing_up',
         'stochastic_continuous_point_mass', 'stochastic_continuous_double_integrator_parking')
SEEDS = tuple(range(10))
MODULES = ('actor', 'qf1', 'qf2', 'qf1_target', 'qf2_target')
OPTIMIZERS = ('q_optimizer', 'actor_optimizer', 'a_optimizer')
ARRAYS = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')


def configuration(task, method, seed, tiny=False):
    raw = yaml.safe_load((ROOT / 'external/SMoSE/config/reacher.yml').read_text())
    raw.update(seed=seed, env_id=task, device='cpu', total_timesteps=100000)
    raw['sac']['nonlinear_actor'] = method == 'SAC'
    if tiny:
        raw.update(learning_starts=16, total_timesteps=80)
        raw['sac'].update(buffer_size=512, batch_size=8)
    return raw, SimpleNamespace(**{**raw, 'sac': SimpleNamespace(**raw['sac'])})


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def new_agent(task, method, seed, tiny=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw, cfg = configuration(task, method, seed, tiny)
    env = make_external_env(task, seed)
    agent = setup_sac(cfg, env)
    obs, _ = env.reset(seed=seed)
    return raw, cfg, env, agent, obs


def advance(cfg, env, agent, obs, t):
    if t < cfg.learning_starts:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action = agent.actor.get_action(torch.as_tensor(obs[None]))[0][0].numpy()
    before = env.state.copy()
    after, reward, terminated, truncated, info = env.step(action)
    agent.rb.add(obs[None], after[None], action[None], np.asarray([reward]),
                 np.asarray([terminated]), [info])
    record = [t + 1, *before, float(action[0]), info['executed_action'],
              info['noise_draw_primary'], info['noise_draw_secondary'],
              *env.state, -reward, terminated, truncated,
              bool(info.get('constraint_violation', False))]
    obs = env.reset()[0] if terminated or truncated else after
    if t >= cfg.learning_starts:
        train_sac(cfg, agent)
    return obs, record


def save_checkpoint(path, env, agent, obs, step, seconds):
    rb = agent.rb
    size = rb.buffer_size if rb.full else rb.pos
    state = {
        'modules': {key: getattr(agent, key).state_dict() for key in MODULES},
        'optimizers': {key: getattr(agent, key).state_dict() for key in OPTIMIZERS},
        'log_alpha': agent.log_alpha.detach(), 'counter': agent.counter.copy(),
        'replay': {key: getattr(rb, key)[:size].copy() for key in ARRAYS},
        'replay_pos': rb.pos, 'replay_full': rb.full,
        'env': {key: copy.deepcopy(getattr(env, key)) for key in
                ('state', 'steps', 'calls', 'training_violations')},
        'env_rng': env.rng.bit_generator.state,
        'noise_rng': env.base._transition_rng.bit_generator.state,
        'action_rng': env.action_space.np_random.bit_generator.state,
        'rng': rng_state(), 'obs': obs, 'step': step, 'seconds': seconds,
    }
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, env, agent):
    state = torch.load(path, weights_only=False, map_location='cpu')
    for key in MODULES:
        getattr(agent, key).load_state_dict(state['modules'][key])
    for key in OPTIMIZERS:
        getattr(agent, key).load_state_dict(state['optimizers'][key])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'])
    agent.counter.update(state['counter'])
    for key, value in state['replay'].items():
        getattr(agent.rb, key)[:len(value)] = value
    agent.rb.pos, agent.rb.full = state['replay_pos'], state['replay_full']
    for key, value in state['env'].items():
        setattr(env, key, value)
    env.rng.bit_generator.state = state['env_rng']
    env.base._transition_rng.bit_generator.state = state['noise_rng']
    env.action_space.np_random.bit_generator.state = state['action_rng']
    restore_rng(state['rng'])
    return state['obs'], state['step'], state['seconds']


def evaluation_inputs(task, seed):
    with np.load(PRIOR / f'{task}_{seed}' / 'eval_inputs.npz', allow_pickle=False) as data:
        initial, tape = data['initial'].copy(), data['tape'].copy()
    assert initial.shape == (20, 2) and tape.shape == (20, 120, 2)
    assert np.isfinite(initial).all() and np.isfinite(tape).all()
    return initial, tape


def evaluate(cfg, env, agent, initial, tape, folder, step):
    saved_rng = rng_state()
    actor = (MLPActor if cfg.sac.nonlinear_actor else Actor)(cfg, env)
    actor.load_state_dict(agent.actor.state_dict())
    actor.eval()
    episodes, trace = [], []
    evaluation_env = environment(cfg.env_id, cfg.seed)
    try:
        for episode, initial_state in enumerate(initial):
            state, cost, violation, terminal = initial_state.copy(), 0., False, None
            for t in range(tape.shape[1]):
                with torch.no_grad():
                    action = float(actor.get_action(torch.as_tensor(state[None], dtype=torch.float32), deterministic=True)[0].item())
                result = evaluation_env.step_with_standard_noise(state, action, tape[episode, t])
                trace.append([episode, t, *state, action, result.info['executed_action'],
                              *tape[episode, t], *result.next_state, result.cost, result.done])
                cost += result.cost
                terminal = result.info.get('terminal')
                violation |= bool(result.info.get('constraint_violation', False)) or terminal == 'collision'
                state = result.next_state
                if result.done:
                    break
            strict = terminal in {'goal', 'stopped', 'parked'}
            success = strict or evaluation_env.distance_to_target(state) < .1
            episodes.append(dict(episode=episode, cost=float(cost), success=float(success),
                                 strict_success=float(strict), violation=float(violation),
                                 reach_avoid=float(strict and not violation), steps=t+1,
                                 terminal=terminal or 'horizon'))
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / f'episodes_{step}.csv', episodes)
        np.savez_compressed(folder / f'trace_{step}.npz', trace=np.asarray(trace),
                            columns=np.asarray(['episode', 't', 'state0', 'state1', 'intended',
                            'executed', 'noise0', 'noise1', 'next0', 'next1', 'cost', 'done']))
        return {key: float(np.mean([row[key] for row in episodes])) for key in
                ('cost', 'success', 'strict_success', 'violation', 'reach_avoid', 'steps')}
    finally:
        restore_rng(saved_rng)


def source_hashes():
    paths = list(HERE.glob('*.py')) + list((ROOT / 'sdta_mdp').glob('*.py'))
    paths += [ROOT / 'external/SMoSE/src/sac.py', ROOT / 'external/SMoSE/config/reacher.yml']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def init():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Manifest already exists; do not overwrite an active protocol.')
    jobs = []
    # Interleave tasks so all four interfaces are exercised early in the queue.
    for seed in SEEDS:
        for task in TASKS:
            evaluation_inputs(task, seed)
            (OUT / 'inputs').mkdir(parents=True, exist_ok=True)
            path = OUT / 'inputs' / f'{task}_{seed}.npz'
            source = PRIOR / f'{task}_{seed}' / 'eval_inputs.npz'
            shutil.copyfile(source, path)
            for method in ('SMoSE', 'SAC'):
                training_seed = int(np.random.SeedSequence([20260923, TASKS.index(task), seed, 0]).generate_state(1)[0])
                jobs.append(dict(id=f'{task}_{method}_s{seed}', task=task, method=method,
                                 seed=seed, training_seed=training_seed,
                                 input_sha=sha(path), input_source=str(source.relative_to(ROOT))))
    protected = list((ROOT / 'reference').rglob('*.csv'))
    import importlib.metadata
    dump(OUT / 'manifest.json', dict(stage='fixed-budget main comparison; convergence not established', jobs=jobs,
         steps=100000, checkpoints=[20000, 50000, 100000], warmup=10000,
         expected_jobs=80, expected_evaluation_rows=240, final_checkpoint=100000,
         budget_policy='Equal fixed 100k interaction budget; no selection or extension using formal test results.',
         python=sys.version, dependencies={name: importlib.metadata.version(name) for name in
             ('torch', 'numpy', 'gymnasium', 'stable-baselines3', 'PyYAML')},
         protected_files={str(p.relative_to(ROOT)): sha(p) for p in protected},
         author_commit='ae2a1a875193bf121ef1b35038994a8d899343b7',
         sources=source_hashes(), noise_model='clipped Gaussian action noise',
         noise_std_action_range_fraction=.05, paper_modified=False))


def worker(job_id):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert source_hashes() == manifest['sources'], 'Frozen source changed'
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    folder = OUT / 'jobs' / job_id
    folder.mkdir(parents=True, exist_ok=True)
    raw, cfg, env, agent, obs = new_agent(job['task'], job['method'], job['training_seed'])
    dump(folder / 'config.json', {**raw, **job})
    input_path = OUT / 'inputs' / f"{job['task']}_{job['seed']}.npz"
    assert sha(input_path) == job['input_sha']
    with np.load(input_path) as data:
        initial, tape = data['initial'], data['tape']
    checkpoint = folder / 'resume.pt'
    start, previous = 0, 0.
    if checkpoint.exists():
        obs, start, previous = load_checkpoint(checkpoint, env, agent)
    begun, records = time.perf_counter(), []
    for t in range(start, cfg.total_timesteps):
        obs, record = advance(cfg, env, agent, obs, t)
        records.append(record)
        step = t + 1
        elapsed = previous + time.perf_counter() - begun
        if step % 1000 == 0:
            assert all(torch.isfinite(p).all() for name in MODULES for p in getattr(agent, name).parameters())
            dump(folder / 'status.json', dict(status='running', step=step, total=cfg.total_timesteps,
                 gradient_updates=agent.counter['n_steps'], elapsed_seconds=elapsed, pid=os.getpid()))
            print(f'{job_id}: {step}/100000 updates={agent.counter["n_steps"]}', flush=True)
        if step in manifest['checkpoints']:
            evaluation_start = time.perf_counter()
            metrics = evaluate(cfg, env, agent, initial, tape, folder, step)
            row = {**job, **metrics, 'training_steps': step, 'training_interactions': env.calls,
                   'gradient_updates': agent.counter['n_steps'], 'elapsed_seconds': elapsed,
                   'evaluation_seconds': time.perf_counter() - evaluation_start,
                   'evaluation_interactions': int(round(metrics['steps'] * len(initial))),
                   'training_violation_steps': env.training_violations,
                   'online_model_predictions': 0, 'noise_model': manifest['noise_model'],
                   'noise_std_action_range_fraction': .05,
                   'stage': manifest['stage'], 'actor_parameters': sum(p.numel() for p in agent.actor.parameters())}
            dump(folder / f'result_{step}.json', row)
        if step % 5000 == 0:
            np.savez_compressed(folder / f'training_{step}.npz', trace=np.asarray(records),
                                columns=np.asarray(['step', 'state0', 'state1', 'intended',
                                'executed', 'noise0', 'noise1', 'next0', 'next1', 'cost',
                                'terminated', 'truncated', 'violation']))
            records.clear()
            save_checkpoint(checkpoint, env, agent, obs, step, previous + time.perf_counter() - begun)
    dump(folder / 'status.json', dict(status='complete', step=cfg.total_timesteps,
         gradient_updates=agent.counter['n_steps'], elapsed_seconds=previous+time.perf_counter()-begun))


def collect(active=None):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((OUT / 'jobs').glob('*/result_*.json'))]
    write_csv(OUT / 'learning_curve.csv', rows)
    states = {}
    for job in manifest['jobs']:
        path = OUT / 'jobs' / job['id'] / 'status.json'
        states[job['id']] = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
    result = dict(jobs=states, completed=sum(s['status'] == 'complete' for s in states.values()),
                  total=len(manifest['jobs']), evaluation_rows=len(rows), active=active or {}, paper_modified=False)
    dump(OUT / 'status.json', result)
    return result


def supervise(workers):
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'supervisor.lock').open('a+b') as lock:
        lock.seek(0)
        if lock.read(1) == b'':
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        jobs = json.loads((OUT / 'manifest.json').read_text())['jobs']
        state = collect()
        pending = [j for j in jobs if state['jobs'][j['id']]['status'] != 'complete']
        active = {}
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                folder = OUT / 'jobs' / job['id']
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'worker.log').open('a', encoding='utf-8')
                proc = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(__file__),
                    'worker', '--job', job['id']], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                active[job['id']] = (proc, log)
            for key, (proc, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    if code:
                        dump(OUT / 'jobs' / key / 'status.json', dict(status='failed', exit_code=code))
                    del active[key]
            collect({key: proc.pid for key, (proc, _) in active.items()})
            if pending or active:
                time.sleep(10)
        with (OUT / 'audit.log').open('a', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(HERE / 'analyze.py')],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        dump(OUT / 'analysis_status.json', dict(exit_code=result.returncode,
                                               status='complete' if result.returncode == 0 else 'failed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'worker', 'supervise', 'status'])
    parser.add_argument('--job')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'init':
        init()
    elif args.command == 'worker':
        import msvcrt
        folder = OUT / 'jobs' / args.job
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'worker.lock').open('a+b') as lock:
            lock.seek(0)
            if lock.read(1) == b'':
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            worker(args.job)
    elif args.command == 'supervise':
        supervise(args.workers)
    else:
        print(json.dumps(collect(), indent=2))
