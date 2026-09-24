# SDTA-MDP

Anonymous code and experiment artifact for **SDTA-MDP: Symbolic Distributed Time Aggregation for Continuous Control**.

This repository contains the algorithm, the seven stochastic benchmarks, the
baselines and analyses used in the accompanying manuscript, and the recorded
results underlying its tables. It is a curated source snapshot with anonymous release updates. Manuscript sources, internal reviews, tests, pilot studies, and
experiments outside the manuscript are excluded.

## Install

Python 3.12 is the recorded environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell's standard activation command.
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

On Windows, `py` can replace `python`. The core algorithm needs NumPy and Z3.
Matplotlib is used only for the optional experiment figure. All reported
experiments ran on CPU. SAC and SMoSE additionally require PyTorch; see below.

## Reproduce the recorded tables

```bash
python scripts/summarize_paper.py
# Optional vector plot of the two main cost/model-call comparisons:
python scripts/summarize_paper.py --figure
```

Outputs go to `generated/`. The supplied main records contain exactly **390**
records: four methods on seven tasks, SymPar+Q on three finite-action tasks,
and SAC and SMoSE on four continuous-action tasks, all over ten seeds. The lookahead data contain **80**
entries: two variants on four continuous-action tasks, over ten seeds.

The main CSV intentionally preserves the paper's provenance: control statistics
and running times are the original measurements; SDTA model-call totals come
from the complete-count replay. SymPar+Q counts exclude actual evaluation
transitions. See [reproducibility details](docs/REPRODUCIBILITY.md).

## Run fresh experiments

The default protocol uses ten seeds, twenty evaluation episodes, 32 absorption
samples, horizon 80, the sampled frontier backend, action slip 0.10 for finite
actions, and clipped Gaussian actuator noise of 5% of the action range for
continuous actions. Noise projection uses three representatives within two
standard deviations. The configuration is in `configs/paper.json`.

```bash
# POSIX shells: export PYTHONHASHSEED=0
# PowerShell:  $env:PYTHONHASHSEED = '0'
python scripts/run_paper_experiments.py --suite main
python scripts/run_paper_experiments.py --suite lookahead
python scripts/run_partition_analysis.py
```

Main execution produces 280 fresh records for SDTA-MDP, MPC, FineGridVI, and
TileQ. SymPar+Q is run separately using the original authors' Scala backend:

```bash
python scripts/run_sympar.py --upstream-root external/SymPar
```

Download and extract the official artifact linked in
[third-party dependencies](THIRD_PARTY.md) first, and install Scala CLI with a
compatible JDK. The runner defaults to the paper's 20,000 training episodes,
gamma 1.0, control-refined supplied partitions, and three finite-action tasks.
It does not rerun JPF/SPF partition discovery.

To run a subset or resume an interrupted run:

```bash
python scripts/run_paper_experiments.py --suite main --benchmarks stochastic_braking_car --seeds 0 --out-dir runs/braking-seed0
python scripts/run_paper_experiments.py --suite main --benchmarks stochastic_braking_car --seeds 0 --out-dir runs/braking-seed0 --resume
```

Resume requires an identical protocol and source manifest. Use a new output
directory after changing code or settings. Fresh results are written under
`runs/` and never overwrite the recorded `reference/` data.

## Code map

| Component | Implementation |
|---|---|
| Executable transition/cost interfaces and seven task definitions | `environment.py`, `continuous_environments.py`, `benchmarks.py` |
| Execution noise and disturbance-projected path predicates | `stochastic_environments.py` |
| SMT-pruned path-signature blocks and frontier support | `symbolic.py` |
| Stopped absorption rollouts: cost, holding time, successor distribution | `absorption.py` |
| Frontier SMDP solve and local controller | `solver.py` |
| Finite/continuous action representations and optional exact safe sets | `actions.py`, `exact_actions.py` |
| Complete preparation/online dynamics-query counter | `model_calls.py` |
| MPC, FineGridVI, and TileQ | `baselines.py` |
| Shared evaluation and the manuscript experiment loop | `evaluation.py`, `paper_experiments.py` |

All module paths in this table are relative to `sdta_mdp/`. The deterministic
base environments remain because the stochastic tasks wrap them; they are not
additional experiment suites. Details of the retained scope are in
[SCOPE.md](docs/SCOPE.md).

## Interpretation and validation

SDTA-MDP is an interpretable approximation pipeline. The implementation uses
least-squares stationary-distribution/Poisson solves and stops when block
actions stabilize. It does not certify a common optimal average-cost gain,
zero Bellman residual, or global optimality for arbitrary tasks.

The release was checked by comparing retained computational definitions with
the source snapshot, reconstructing manuscript table values, and running small
execution checks. These checks establish package integrity; they are not a new
ten-seed replication of every experiment. See
[VALIDATION.md](docs/VALIDATION.md) for the exact verification scope.

## SAC and SMoSE comparisons

The release includes 80 frozen final actors, 40 main evaluation input sets,
per-episode results, and the training and evaluation adapters. To reproduce the
learned-policy results without retraining:

```bash
python -m pip install -r requirements-learned.txt
python scripts/setup_learned.py
python scripts/learned/evaluate_recorded.py
```

For a fresh fixed-budget training run (the supervisor uses Windows file locks):

```powershell
python scripts/learned/run.py init
python scripts/learned/run.py supervise --workers 2
python scripts/learned/evaluate_recorded.py --checkpoint-dir runs/learned/jobs
```

Each run uses 100,000 interactions, including 10,000 random warmup actions.
The main comparison always uses the final policy. Training diagnostics use a
separate archived input set; `evaluate_recorded.py` applies the original main
experiment reset protocol. Read [the learned-baseline protocol](docs/LEARNED_BASELINES.md).
