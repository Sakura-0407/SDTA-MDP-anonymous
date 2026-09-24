# Protocol and data provenance

## Recorded data

- `reference/main.csv`: 390 observations (310 prior records plus 80 SAC/SMoSE records), with model
  counts assembled using the exact convention used in the manuscript.
- `reference/sdta_complete_counts.csv`: the 70 SDTA replay records used for
  complete preparation plus online model-call totals.
- `reference/lookahead_ablation.csv`: 40 full-method and 40 NoLookahead records;
  these retain the original measured controls and timings.
- `reference/partition_sizes.csv`: seven task summaries, including independent
  block reconstruction and measured first-exit edge counts.
- `reference/sympar/*.json`: the three supplied-partition manifests, including
  training settings, partition witnesses, and upstream artifact hashes.

The main CSV keeps `recorded_model_calls` for provenance. For SDTA,
`model_calls = preparation_model_calls + online_model_calls` uses the replay;
for SymPar+Q, `model_calls = training_model_calls`. MPC, FineGridVI, and TileQ
retain their recorded planning/training counts. Actual evaluation transitions
are excluded. Control metrics and runtime columns are not silently replaced by
replay measurements.

The replay did not reproduce every historical measurement exactly. In
ContinuousBrakingCar, seeds 0-4 showed differences in historical absorption/edge
records; costs differed for seeds 1 and 3 by approximately -0.000025 and
-0.1249625. Success and violation rates matched. The release therefore keeps
the historical controls and labels the replay source of the counts explicitly.
Do not interpret the supplied CSV as one new simultaneous run.

## Fresh runs

The Python experiment entrypoint retains the existing planning/execution
separation and seed-matched disturbance protocol. Model instrumentation counts
leaf dynamics calls during preparation and online planning. Actual evaluation
transitions and figure generation are outside that counter.

The paper's principal settings are:

| Setting | Value |
|---|---|
| Seeds | 0-9 |
| Evaluation episodes per seed | 20 |
| Maximum evaluation length | Environment default, 120 steps |
| Absorption samples / horizon | 32 / 80 |
| Main frontier backend | Sampled |
| Finite action slip | 0.10, uniformly to another action |
| Continuous actuator noise | 0.05 of action-range width, clipped Gaussian |
| Projected continuous noise support | 3 representatives within 2 standard deviations |
| FineGridVI | 8 bins/dimension; discount 0.98 |
| FineGrid continuous expectation | 7-point Gauss-Hermite quadrature |
| MPC | Horizon 6, 8 common scenarios per candidate |
| TileQ | 4 bins/dimension and 400 training episodes at the full protocol |
| SymPar+Q | 20,000 episodes, gamma 1.0, supplied control-refined partitions |

Local candidate construction and task-specific policy routines are retained
from the original environment definitions. No global controller rewrite is
introduced by this packaging step.

`--episodes`, `--absorption-samples`, and `--absorption-horizon` permit short
execution checks; non-default values are not the paper protocol. SymPar's
training-episode override likewise permits checking the adapter and Scala
installation without running the full study.

Use Python 3.12.10 and `requirements.txt` for the recorded dependency versions.
The recorded CPU system was Windows 11 Pro, Intel Core i5-10300H, 16 GB RAM.
Different platforms, solver versions, SMT witness enumeration, and process
state can affect fresh results and timing. A small package check does not
establish statistical equivalence to the full historical run.

## Statistics and partition scaling

The summarizer uses the retained bootstrap implementation: 1,000 resamples of
seed-level means, RNG seed 2026, percentile 95% intervals. Paired SDTA/MPC
comparisons use 10,000 random sign flips over matched seeds. Scaling ratios
are averaged per task before averaging across the three finite or four
continuous tasks; dividing the group mean cell count by group mean block count
is a different statistic.

`run_partition_analysis.py` independently constructs the noise-aware blocks.
It does not invent empirical first-exit edges; those require absorption runs.
Reference geometric scaling and full main control/runtime summaries can be
regenerated without rerunning the expensive experiments.

For SAC and SMoSE training, frozen actors, and the main evaluation protocol, see [LEARNED_BASELINES.md](LEARNED_BASELINES.md).
