# SAC and SMoSE reproducibility

The continuous-action comparison adds two methods on four tasks, ten seeds
per method and task: 80 trained policies and 1,600 final evaluation episodes.
The primary results are `reference/learned/results.csv`, also included in the
390 records in `reference/main.csv`. Existing planning baseline measurements
are preserved.

## Training

Both methods use the original SMoSE author's SAC implementation and Reacher
configuration, fetched by `scripts/setup_learned.py`. SAC uses the full
256-by-256 MLP actor; SMoSE uses eight experts with top-one routing. Each run
has 100,000 interactions including 10,000 random warmup actions. The final
100,000-step policy is used without selecting a checkpoint on evaluation
results. Equal budgets do not establish convergence.

The reward is negative task cost, without reward shaping or observation
normalization. True terminations stop bootstrapping; truncation at 120 steps
retains bootstrapping. Training random streams are separate from evaluation.
The recorded dependency versions are in `requirements-learned.txt`.

`scripts/learned/run.py init` creates a new protocol manifest under
`runs/learned`. `supervise --workers 2` starts or resumes the Windows training
queue. Its file locks prevent duplicate jobs. Full optimizer, replay, environment,
and RNG checkpoints are written every 5,000 steps. The source hashes must
match the initialized protocol to resume. The public actor files contain only
the final policy tensors and provenance; they are for evaluation, not resuming
training. Fresh training creates the full resumable checkpoints.

Training diagnostics at 20k, 50k, and 100k use the archived diagnostic inputs
in `reference/learned/training_inputs`. These diagnostic metrics are not the
main table. After training, run `evaluate_recorded.py --checkpoint-dir
runs/learned/jobs` to evaluate the final policies under the main protocol.

## Main evaluation

`scripts/learned/evaluate_recorded.py` reconstructs the original main reset
protocol: twenty resets from `default_rng(seed)` and an independent execution
noise stream, with a maximum of 120 transitions per episode. The deterministic
actor output is perturbed by the same clipped Gaussian actuator noise as the
other continuous methods, with standard deviation 5% of the action-range width.

The script compares the reconstructed inputs with the 40 archived main input
sets, checks results against the original `evaluate_policy` implementation,
and, when using the supplied actors, checks all 80 seed-level results against
the recorded metrics. Per-episode records are provided in
`reference/learned/episodes`. New evaluation outputs go under `runs/`.

## Resource accounting

Training interactions are recorded separately as 100,000 for each run. They
are included in the main CSV's `model_calls` to match the manuscript's Calls
comparison. Policy evaluation uses zero online model predictions. Actual
environment transitions during evaluation are excluded from model-call totals.

The elapsed time is the original training-run measurement, including diagnostic
and checkpoint overhead. It is not relabeled as pure optimizer time. The
learned-only summary provides sample standard deviations and percentile 95%
bootstrap intervals from 10,000 resamples with seed 20260924. The general
repository summarizer retains its previously documented bootstrap convention.
