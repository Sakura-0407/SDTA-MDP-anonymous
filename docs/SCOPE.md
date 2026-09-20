# Manuscript scope

## Retained benchmarks

| Finite actions | Bounded continuous actions |
|---|---|
| BrakingCar | ContinuousBrakingCar |
| PointMass | Pendulum |
| DoubleIntegrator | Continuous PointMass |
| | Continuous DoubleIntegrator |

Each published benchmark is exposed as `stochastic_<base_name>`. Its underlying
deterministic transition implementation is retained as a dependency. The paper
runner accepts only the seven stochastic tasks.

## Retained evidence

- Main stochastic control comparison: SDTA-MDP, MPC, FineGridVI, TileQ, and
  upstream SymPar+Q on the three applicable finite-action tasks.
- Local lookahead ablation on the four continuous-action tasks.
- Symbolic block sizes, independent partition reconstruction, empirical
  first-exit edge counts, and geometric partition scaling.
- Recorded preparation/evaluation running times.
- Complete model-query accounting and the data needed to reconstruct the
  manuscript's cost, success, violation, and call-count summaries.

## Excluded material

Unit/integration tests, smoke outputs, internal manuscript audits, revision
history, paper LaTeX/PDF files, translations, private notes, IDE configuration,
virtual environments, caches, logs, checkpoints, and the original Git history
are not part of this release.

The registry and environment modules exclude both MountainCar tasks, separate
wind-transition tasks, and large-maze scaling tasks. Deep-RL implementations,
strong-MPC diagnostics, internal Python SymPar-Q approximations, standalone
random-policy/grid-partition control experiments, broad legacy launchers, and
later exploratory studies are also excluded.

Shared algorithm capabilities described by the method, including exact-LRA
frontier discovery and optional exact safe-action sets, remain in the core
modules. The published main experiment explicitly selects sampled frontiers.

## Computational preservation

The retained solver, partitioner, absorption analyzer, stochastic wrapper,
action definitions, and model-call counter are unchanged. Unused environment
classes and baseline implementations were removed as whole definitions.
Retained environment/baseline definitions and the SDTA evaluation function have
the same syntax trees as the original source. The experiment configuration and
entrypoints were narrowed to the manuscript scope.

The SymPar adapter was narrowed to three tasks and its defaults set to the
recorded protocol. Its reported `model_calls` is the training count, with
evaluation-transition counts retained in a separate column, matching the
manuscript's accounting convention.
