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
