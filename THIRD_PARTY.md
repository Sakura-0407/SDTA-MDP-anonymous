# Third-party dependencies

The Python package depends on NumPy and Z3. Matplotlib is an optional plotting
dependency. Dependencies retain their own licenses and attribution.

The SymPar+Q comparison uses the original authors' Q-learning implementation
from the official **Symbolic State Partitioning for Reinforcement Learning**
artifact: <https://doi.org/10.5281/zenodo.14620119>.

This repository includes the supplied-partition adapter, but does not
redistribute the upstream binary or the upstream project's unrelated
experiments. Download the artifact and point `--upstream-root` at its `SymPar`
directory, which must contain:

```text
SymPar/
  symsim-files/
    symsim.jar
    project.scala
```

The recorded upstream files have these SHA-256 digests:

```text
symsim.jar    7f1de2124ecadf71e130d16c0105f58775278eeeeeead1beec6a505ed99908b9
project.scala 304fa558b84962792af06fd308ccc425a42837c0860458818aa2dc729da38eff
```

Keep the upstream artifact's `LICENSE.txt` and attribution with any local copy.
The adapter uses Scala CLI, Scala 3.3.0, and the dependency declarations in
`project.scala`. JPF/SPF discovery is excluded from this particular comparison;
the control-refined partitions are supplied by the adapter.


## SAC and SMoSE

Both baselines use the original [SMoSE repository](https://github.com/vinczematyas/SMoSE),
revision `ae2a1a875193bf121ef1b35038994a8d899343b7`, by Matyas Vincze and coauthors.
`scripts/setup_learned.py` obtains and checksum-verifies `src/sac.py` and
`config/reacher.yml` from that repository; these third-party source files are
not redistributed here. PyTorch, Gymnasium, Stable-Baselines3, and PyYAML retain
their respective licenses. The local training and environment adapters use
the unmodified upstream actor, critic, and SAC update routines.
