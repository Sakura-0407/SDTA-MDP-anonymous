from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sdta_mdp.benchmarks import make_benchmark


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def environment(task, seed=0, points=3):
    env = make_benchmark(task)
    env.slip_probability = .10
    env.continuous_noise_fraction = .05
    env.project_stochastic_paths = True
    env.gaussian_support_sigma = 2.
    env.gaussian_support_points = points
    env.set_transition_seed(seed)
    return env


def make_external_env(task, seed):
    import gymnasium as gym
    from gymnasium import spaces

    class Wrapper(gym.Env):
        def __init__(self):
            self.base = environment(task, seed)
            low, high = self.base.action_bounds()
            self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
            self.observation_space = spaces.Box(self.base.bounds.low.astype(np.float32), self.base.bounds.high.astype(np.float32), dtype=np.float32)
            self.single_action_space = self.action_space
            self.single_observation_space = self.observation_space
            self.action_space.seed(seed)
            self.rng = np.random.default_rng(seed)
            self.calls, self.steps, self.training_violations = 0, 0, 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.state = self.base.reset(self.rng)
            self.steps = 0
            return self.state.astype(np.float32), {}

        def step(self, action):
            step = self.base.step(self.state, float(np.asarray(action).reshape(-1)[0]))
            self.state = step.next_state
            self.steps += 1
            self.calls += 1
            self.training_violations += int(step.info.get("constraint_violation", False))
            return self.state.astype(np.float32), -float(step.cost), bool(step.done), bool(self.steps >= 120 and not step.done), step.info

    return Wrapper()
