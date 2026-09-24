# Release validation

Validation was performed on the isolated publication snapshot, without changing
the development checkout.

- Retained computational class/function definitions were compared with the
  source snapshot. The algorithm and retained environment/baseline definitions
  were preserved; the evaluation configuration, imports, benchmark registry,
  and experiment entrypoints were narrowed to the manuscript scope.
- The eight algorithm/support files copied without pruning were verified
  byte-for-byte against the source snapshot.
- The reference dataset contains 310 unique main task/method/seed records,
  80 lookahead records, 70 complete-count replay records, and seven partition
  summaries. A comparison of 186 formatted cost, confidence-interval, success,
  violation, model-call, runtime, and ablation cells found no mismatch with the
  accompanying manuscript.
- Small execution checks completed for SDTA-MDP on all seven tasks; MPC,
  FineGridVI, and TileQ on all seven tasks; NoLookahead on the four continuous
  tasks; and the original Scala SymPar+Q backend on the three finite tasks.
  These comprise 35 task/method runs, with one evaluation episode and seed 0.
  SDTA used two absorption samples and horizon five; SymPar used two training
  episodes. The other settings retained the runner defaults.
- Independent partition reconstruction on stochastic BrakingCar produced six
  blocks. The table summarizer and its optional vector-PDF export completed.
- Publication files were checked for local absolute paths, account identifiers,
  credentials, and development-only artifacts. The repository starts with a
  new anonymous commit rather than the development history.

The execution checks establish that the retained entrypoints and external
adapter run. They do **not** constitute a fresh full-protocol, ten-seed
replication or establish the solver's theoretical guarantees. Historical
replay differences and counting conventions are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Internal checking scripts, temporary logs, and smoke-run outputs are deliberately
not part of this publication repository.

## SAC and SMoSE release update

The added 80 final policies were evaluated over 1,600 episodes. All 80 results
matched the recorded metrics and the original evaluator. The 40 reconstructed
main input sets matched the archived inputs exactly. All 32 numeric cells in
the current continuous-action main table were reconstructed from the 390-row
CSV. The prior 310 observations were preserved field-for-field.

Five execution checks passed: checkpoint/resume for both methods, evaluation
RNG isolation and transition replay, fixed paired inputs, all eight task/method
interfaces, and all forty diagnostic input sets. The training, checkpoint, and
evaluation kernels were compared structurally with the experiment source.
These checks do not constitute a fresh 100,000-step retraining of 80 policies.
