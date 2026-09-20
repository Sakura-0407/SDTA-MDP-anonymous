from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import z3

from .environment import ActionPathSupportMap, AffineSuccessorPiece, ContinuousEnvironment, SymbolicPathCondition


@dataclass(frozen=True)
class SymbolicBlock:
    """由跨 action 路径条件求交得到的连续状态符号块。"""

    block_id: int
    constraints: tuple[z3.BoolRef, ...]
    action_paths: ActionPathSupportMap
    atom_values: Mapping[str, bool]
    descriptions: tuple[str, ...]
    witness: np.ndarray
    role: str
    frontier_boundaries: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    def formula(self) -> z3.BoolRef:
        if not self.constraints:
            return z3.BoolVal(True)
        return z3.And(*self.constraints)

    @property
    def is_truncated(self) -> bool:
        return self.role in {"edge", "remote"}


@dataclass(frozen=True)
class FrontierTransition:
    """一次抽样发现的跨符号块前沿跃迁。"""

    source_block: int
    action: float
    target_block: int
    boundary: str
    origin: str = "unknown"


@dataclass(frozen=True)
class PartitionReport:
    """符号分区与前沿发现的输出。"""

    blocks: tuple[SymbolicBlock, ...]
    path_counts: Mapping[float, int]
    frontier_transitions: tuple[FrontierTransition, ...]
    frontier_mode: str = "auto"
    frontier_backend: str = "unknown"

    @property
    def core_blocks(self) -> tuple[SymbolicBlock, ...]:
        return tuple(block for block in self.blocks if block.role == "core")

    @property
    def truncated_blocks(self) -> tuple[SymbolicBlock, ...]:
        return tuple(block for block in self.blocks if block.is_truncated)


@dataclass(frozen=True)
class _Candidate:
    constraints: tuple[z3.BoolRef, ...]
    action_paths: ActionPathSupportMap
    atom_values: Mapping[str, bool]
    descriptions: tuple[str, ...]

    def formula(self) -> z3.BoolRef:
        if not self.constraints:
            return z3.BoolVal(True)
        return z3.And(*self.constraints)

    def combine(self, other: "_Candidate") -> "_Candidate | None":
        merged_atoms = dict(self.atom_values)
        for name, truth in other.atom_values.items():
            if name in merged_atoms and merged_atoms[name] != truth:
                return None
            merged_atoms[name] = truth
        return _Candidate(
            constraints=(*self.constraints, *other.constraints),
            action_paths={**self.action_paths, **other.action_paths},
            atom_values=merged_atoms,
            descriptions=(*self.descriptions, *other.descriptions),
        )


class SymbolicPartitioner:
    """SymPar 风格的 Z3 符号分区器。

    旧 SymPar 项目先对每个 action 运行 Symbolic PathFinder 得到 `PC^a`，
    再逐层求交并用 SMT solver 剪枝。这里把白盒 Python 环境直接暴露出的
    单步路径条件当成同等输入，因此可以在连续实数状态空间上复用这条主线。
    """

    def __init__(
        self,
        env: ContinuousEnvironment,
        *,
        timeout_ms: int = 2_000,
        frontier_probe_samples: int = 8,
        frontier_mode: str = "auto",
        seed: int = 7,
    ) -> None:
        frontier_mode = "exact-lra" if frontier_mode == "exact" else frontier_mode
        if frontier_mode not in {"auto", "exact-lra", "sampled"}:
            raise ValueError("frontier_mode must be one of: auto, exact-lra, sampled")
        self.env = env
        self.timeout_ms = timeout_ms
        self.frontier_probe_samples = frontier_probe_samples
        self.frontier_mode = frontier_mode
        self.rng = np.random.default_rng(seed)
        self.variables = env.make_symbolic_variables()
        self._sat_cache: dict[str, z3.CheckSatResult] = {}
        self._last_blocks: tuple[SymbolicBlock, ...] = ()
        self._last_frontier_backend = "unknown"

    def discover(self) -> PartitionReport:
        blocks, path_counts = self.build_partition()
        transitions = self.discover_frontiers(blocks)
        return PartitionReport(
            blocks=blocks,
            path_counts=path_counts,
            frontier_transitions=transitions,
            frontier_mode=self.frontier_mode,
            frontier_backend=self._last_frontier_backend,
        )

    def build_partition(self) -> tuple[tuple[SymbolicBlock, ...], Mapping[float, int]]:
        """Construct only the SymPar-style semantic partition.

        Frontier support is intentionally excluded so partition-only baselines
        can reuse the same blocks without paying for SDTA-MDP's successor
        analysis.
        """

        if getattr(self.env, "projected_paths_shared_across_actions", lambda: False)():
            action = float(self.env.actions[0])
            shared = self._atomize_action_paths(
                action,
                self.env.symbolic_path_conditions(action),
            )
            candidates = [
                _Candidate(
                    constraints=candidate.constraints,
                    action_paths={
                        float(item): tuple(candidate.action_paths[action])
                        for item in self.env.actions
                    },
                    atom_values=candidate.atom_values,
                    descriptions=candidate.descriptions,
                )
                for candidate in shared
            ]
            path_counts = {
                float(item): len(candidates)
                for item in self.env.actions
            }
            blocks = tuple(
                self._candidate_to_block(idx, candidate)
                for idx, candidate in enumerate(candidates)
            )
            self._last_blocks = blocks
            return blocks, path_counts

        if self.env.path_conditions_may_overlap():
            candidates: list[_Candidate] = [
                _Candidate(
                    constraints=(z3.BoolVal(True),),
                    action_paths={},
                    atom_values={},
                    descriptions=(),
                )
            ]
            path_counts: dict[float, int] = {}
            for action_value in self.env.actions:
                action = float(action_value)
                candidates = self._refine_overlapping_action(
                    candidates,
                    action,
                    self.env.symbolic_path_conditions(action),
                )
                path_counts[action] = len(
                    {candidate.action_paths[action] for candidate in candidates}
                )
            blocks = tuple(
                self._candidate_to_block(idx, candidate)
                for idx, candidate in enumerate(candidates)
            )
            self._last_blocks = blocks
            return blocks, path_counts

        layers: list[tuple[_Candidate, ...]] = []
        path_counts: dict[float, int] = {}
        for action in self.env.actions:
            path_conditions = self.env.symbolic_path_conditions(float(action))
            if self.env.path_conditions_may_overlap():
                candidates = self._atomize_action_paths(float(action), path_conditions)
            else:
                candidates = []
                for pc in path_conditions:
                    candidate = _Candidate(
                        constraints=pc.constraints,
                        action_paths={float(action): (pc.name,)},
                        atom_values=dict(pc.atom_values),
                        descriptions=(pc.description,),
                    )
                    if self._is_sat(candidate.constraints):
                        candidates.append(candidate)
            layers.append(tuple(candidates))
            path_counts[float(action)] = len(candidates)

        combined = self._intersect_layers(layers)
        blocks = tuple(self._candidate_to_block(idx, candidate) for idx, candidate in enumerate(combined))
        self._last_blocks = blocks
        return blocks, path_counts

    def _refine_overlapping_action(
        self,
        candidates: Sequence[_Candidate],
        action: float,
        path_conditions: Sequence[SymbolicPathCondition],
    ) -> list[_Candidate]:
        cells: list[tuple[z3.BoolRef, ActionPathSupportMap, tuple[str, ...], tuple[str, ...]]] = [
            (candidate.formula(), candidate.action_paths, candidate.descriptions, ())
            for candidate in candidates
        ]
        for path in path_conditions:
            path_formula = z3.And(*path.constraints) if path.constraints else z3.BoolVal(True)
            next_cells: list[
                tuple[z3.BoolRef, ActionPathSupportMap, tuple[str, ...], tuple[str, ...]]
            ] = []
            for cell_formula, action_paths, descriptions, support in cells:
                positive = z3.And(cell_formula, path_formula)
                negative = z3.And(cell_formula, z3.Not(path_formula))
                if self._is_sat((positive,)):
                    next_cells.append(
                        (positive, action_paths, descriptions, (*support, path.name))
                    )
                if self._is_sat((negative,)):
                    next_cells.append((negative, action_paths, descriptions, support))
            cells = next_cells

        grouped: dict[
            tuple[tuple[float, tuple[str, ...]], ...],
            tuple[dict[float, tuple[str, ...]], list[z3.BoolRef], tuple[str, ...]],
        ] = {}
        for formula, existing_paths, descriptions, support in cells:
            if not support:
                raise RuntimeError(
                    f"Projected path conditions for action {action:g} do not cover the bounded state space."
                )
            merged_paths = {
                **existing_paths,
                action: tuple(sorted(support)),
            }
            key = tuple(sorted(merged_paths.items()))
            if key not in grouped:
                grouped[key] = (merged_paths, [], descriptions)
            grouped[key][1].append(formula)

        refined: list[_Candidate] = []
        for merged_paths, formulas, descriptions in grouped.values():
            support = merged_paths[action]
            refined.append(
                _Candidate(
                    constraints=(z3.simplify(z3.Or(*formulas)),),
                    action_paths=merged_paths,
                    atom_values={},
                    descriptions=(
                        *descriptions,
                        f"possible paths under action {action:g}: {', '.join(support)}",
                    ),
                )
            )
        return refined

    def _atomize_action_paths(
        self,
        action: float,
        path_conditions: Sequence[SymbolicPathCondition],
    ) -> list[_Candidate]:
        cells: list[tuple[z3.BoolRef, tuple[str, ...]]] = [(z3.BoolVal(True), ())]
        for path in path_conditions:
            path_formula = z3.And(*path.constraints) if path.constraints else z3.BoolVal(True)
            next_cells: list[tuple[z3.BoolRef, tuple[str, ...]]] = []
            for cell_formula, support in cells:
                positive = z3.And(cell_formula, path_formula)
                negative = z3.And(cell_formula, z3.Not(path_formula))
                if self._is_sat((positive,)):
                    next_cells.append((positive, (*support, path.name)))
                if self._is_sat((negative,)):
                    next_cells.append((negative, support))
            cells = next_cells

        grouped: dict[tuple[str, ...], list[z3.BoolRef]] = {}
        for formula, support in cells:
            if not support:
                raise RuntimeError(
                    f"Projected path conditions for action {action:g} do not cover the bounded state space."
                )
            grouped.setdefault(tuple(sorted(support)), []).append(formula)

        candidates: list[_Candidate] = []
        for support, formulas in grouped.items():
            formula = z3.simplify(z3.Or(*formulas))
            candidates.append(
                _Candidate(
                    constraints=(formula,),
                    action_paths={action: support},
                    atom_values={},
                    descriptions=(f"possible paths under action {action:g}: {', '.join(support)}",),
                )
            )
        return candidates

    def _intersect_layers(self, layers: Sequence[Sequence[_Candidate]]) -> tuple[_Candidate, ...]:
        if not layers:
            return ()

        result = list(layers[0])
        for layer in layers[1:]:
            next_result: list[_Candidate] = []
            for current in result:
                for candidate in layer:
                    intersection = current.combine(candidate)
                    if intersection is None:
                        continue
                    if self._is_sat(intersection.constraints):
                        next_result.append(intersection)
            result = self._dedupe_semantic(next_result)
        return tuple(result)

    def _candidate_to_block(self, block_id: int, candidate: _Candidate) -> SymbolicBlock:
        witness = self._witness(candidate.constraints)
        if witness is None:
            raise RuntimeError("内部错误：不可满足候选不应被转换成符号块。")
        role = self._classify_role(witness, candidate)
        boundaries = self._frontier_boundaries(candidate)
        lower, upper = self._box_from_candidate(candidate)
        return SymbolicBlock(
            block_id=block_id,
            constraints=candidate.constraints,
            action_paths=dict(candidate.action_paths),
            atom_values=dict(candidate.atom_values),
            descriptions=tuple(dict.fromkeys(candidate.descriptions)),
            witness=witness,
            role=role,
            frontier_boundaries=boundaries,
            lower=lower,
            upper=upper,
        )

    def _box_from_candidate(self, candidate: _Candidate) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.env.bounds_from_atom_assignment(candidate.atom_values)
        low, high = self.env.refine_bounds_from_paths(candidate.action_paths, low, high)
        if self.env.path_conditions_may_overlap():
            formula_low, formula_high = self._formula_bounds(candidate.formula())
            low = np.maximum(low, formula_low)
            high = np.minimum(high, formula_high)
        return low, high

    def _formula_bounds(self, formula: z3.BoolRef) -> tuple[np.ndarray, np.ndarray]:
        lower = self.env.bounds.low.copy()
        upper = self.env.bounds.high.copy()
        base_constraints = (*self.env.bound_constraints(self.variables), formula)
        for index, name in enumerate(self.env.state_names):
            variable = self.variables[name]
            lower_value = self._optimize_coordinate(base_constraints, variable, minimize=True)
            upper_value = self._optimize_coordinate(base_constraints, variable, minimize=False)
            if lower_value is not None:
                lower[index] = max(lower[index], lower_value)
            if upper_value is not None:
                upper[index] = min(upper[index], upper_value)
        return lower, upper

    def _optimize_coordinate(
        self,
        constraints: Sequence[z3.BoolRef],
        variable: z3.ArithRef,
        *,
        minimize: bool,
    ) -> float | None:
        optimizer = z3.Optimize()
        optimizer.set(timeout=self.timeout_ms)
        optimizer.add(*constraints)
        handle = optimizer.minimize(variable) if minimize else optimizer.maximize(variable)
        if optimizer.check() != z3.sat:
            return None
        values = optimizer.lower_values(handle) if minimize else optimizer.upper_values(handle)
        if len(values) != 3 or not z3.is_rational_value(values[1]):
            return None
        value = _z3_to_float(values[1])
        if z3.is_rational_value(values[2]):
            epsilon_direction = _z3_to_float(values[2])
            if epsilon_direction != 0.0:
                value += float(np.sign(epsilon_direction)) * 1e-9
        return value

    def _classify_role(self, witness: np.ndarray, candidate: _Candidate) -> str:
        return self.env.block_role(witness, candidate.atom_values, candidate.action_paths)

    def _frontier_boundaries(self, candidate: _Candidate) -> tuple[str, ...]:
        boundaries: list[str] = []
        atom_boundaries = self.env.atom_boundaries()
        for atom_name in candidate.atom_values:
            if atom_name in atom_boundaries:
                boundaries.append(atom_boundaries[atom_name])

        for action, support in candidate.action_paths.items():
            if any(
                path in {"far_stop_in_window", "far_continue", "near_stop_in_window", "near_continue"}
                for path in support
            ):
                boundaries.append(f"velocity = {-action * self.env.dt:.6g}")
        return tuple(dict.fromkeys(boundaries))

    def discover_frontiers(self, blocks: Sequence[SymbolicBlock] | None = None) -> tuple[FrontierTransition, ...]:
        blocks = tuple(blocks or self._last_blocks)
        if self.frontier_mode in {"auto", "exact-lra"}:
            try:
                exact_transitions = self._discover_frontiers_exact_lra(blocks)
            except RuntimeError:
                if self.frontier_mode == "exact-lra":
                    raise
                exact_transitions = None
            if exact_transitions is not None:
                self._last_frontier_backend = "exact-lra"
                return exact_transitions
            if self.frontier_mode == "exact-lra":
                raise RuntimeError(
                    f"{self.env.name} does not expose affine successor pieces for exact-LRA frontier discovery."
                )
        self._last_frontier_backend = "sampled-forced" if self.frontier_mode == "sampled" else "sampled-fallback"
        # Without exact-LRA, the H-truncated absorption rollouts discover the
        # empirical frontier support. Avoid a separate incomplete one-step
        # sampling pass here.
        return ()

    def _discover_frontiers_exact_lra(self, blocks: Sequence[SymbolicBlock]) -> tuple[FrontierTransition, ...] | None:
        action_pieces: dict[float, tuple[AffineSuccessorPiece, ...]] = {}
        for action in self.env.actions:
            pieces = self.env.affine_successor_pieces(self.variables, float(action))
            if not pieces:
                return None
            for piece in pieces:
                self._validate_affine_piece(piece)
            action_pieces[float(action)] = pieces

        transitions: list[FrontierTransition] = []
        seen: set[tuple[int, float, int]] = set()
        for source in blocks:
            for action, pieces in action_pieces.items():
                for piece in pieces:
                    base_constraints = (*source.constraints, *piece.guard)
                    base_solver = self._new_qf_lra_solver(base_constraints)
                    base_result = base_solver.check()
                    if base_result == z3.unknown:
                        base_result = self._sat_result(base_constraints)
                    if base_result == z3.unknown:
                        raise RuntimeError(
                            f"SMT returned unknown while checking exact-LRA piece feasibility for block {source.block_id}."
                        )
                    if base_result != z3.sat:
                        continue
                    successor_exprs = self._piece_successor_expressions(piece)
                    substitutions = tuple(
                        (self.variables[name], successor_exprs[index])
                        for index, name in enumerate(self.env.state_names)
                    )
                    for target in blocks:
                        if target.block_id == source.block_id:
                            continue
                        if not self._affine_image_box_may_overlap(source, target, piece):
                            continue
                        target_after_step = z3.substitute(target.formula(), *substitutions)
                        base_solver.push()
                        base_solver.add(target_after_step)
                        sat_result = base_solver.check()
                        base_solver.pop()
                        if sat_result == z3.unknown:
                            constraints = (*source.constraints, *piece.guard, target_after_step)
                            sat_result = self._sat_result(constraints)
                        if sat_result == z3.unknown:
                            raise RuntimeError(
                                f"SMT returned unknown while checking exact-LRA frontier {source.block_id}->{target.block_id}."
                            )
                        if sat_result != z3.sat:
                            continue
                        key = (source.block_id, action, target.block_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        transitions.append(
                            FrontierTransition(
                                source_block=source.block_id,
                                action=action,
                                target_block=target.block_id,
                                boundary=self._boundary_between(source, target),
                                origin="exact-lra",
                            )
                        )
        return tuple(transitions)

    def block_for_state(
        self,
        state: Sequence[float] | np.ndarray,
        blocks: Sequence[SymbolicBlock] | None = None,
    ) -> SymbolicBlock | None:
        blocks = tuple(blocks or self._last_blocks)
        if not blocks:
            return None
        state_arr = self.env.bounds.clip(state)
        for block in blocks:
            if self.contains(block, state_arr):
                return block
        return None

    def contains(self, block: SymbolicBlock, state: Sequence[float] | np.ndarray) -> bool:
        state_arr = self.env.bounds.clip(state)
        # ``lower``/``upper`` are sampling boxes.  Open predicates are inset by
        # 1e-5 when those boxes are built, so use a slightly larger margin
        # before applying the authoritative symbolic formula.
        sampling_margin = 1.1e-5
        if not bool(
            np.all(state_arr >= block.lower - sampling_margin)
            and np.all(state_arr <= block.upper + sampling_margin)
        ):
            return False
        substitutions = [
            (self.variables[name], z3.RealVal(str(float(state_arr[idx]))))
            for idx, name in enumerate(self.env.state_names)
        ]
        expr = z3.simplify(z3.substitute(block.formula(), *substitutions))
        return z3.is_true(expr)

    def sample_block(self, block: SymbolicBlock, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, len(self.env.state_names)), dtype=float)
        low, high = block.lower.copy(), block.upper.copy()
        if np.any(low > high):
            low, high = self.env.bounds.low, self.env.bounds.high
        samples: list[np.ndarray] = []
        attempts = max(200, n * 120)
        for _ in range(attempts):
            if len(samples) >= n:
                break
            point = self.rng.uniform(low, high)
            if self.contains(block, point):
                samples.append(point)
        if not samples:
            samples.append(block.witness.copy())
        return np.asarray(samples[:n], dtype=float)

    def make_frontier_transition(
        self,
        source: SymbolicBlock,
        action: float,
        target: SymbolicBlock,
        *,
        origin: str,
    ) -> FrontierTransition:
        """Create a consistently labelled frontier edge for downstream reports."""

        return FrontierTransition(
            source_block=source.block_id,
            action=float(action),
            target_block=target.block_id,
            boundary=self._boundary_between(source, target),
            origin=origin,
        )

    def _boundary_between(self, source: SymbolicBlock, target: SymbolicBlock) -> str:
        source_atoms = self.env.evaluate_atoms(source.witness)
        target_atoms = self.env.evaluate_atoms(target.witness)
        boundaries = self.env.atom_boundaries()
        changed = [
            boundaries[name]
            for name, truth in source_atoms.items()
            if name in target_atoms and truth != target_atoms[name] and name in boundaries
        ]
        if changed:
            return "; ".join(changed)
        if source.frontier_boundaries:
            return source.frontier_boundaries[0]
        return "implicit frontier reachability"

    def _validate_affine_piece(self, piece: AffineSuccessorPiece) -> None:
        dim = len(self.env.state_names)
        if piece.matrix.shape != (dim, dim):
            raise RuntimeError(f"Affine successor piece {piece.label} has invalid matrix shape.")
        if piece.offset.shape != (dim,):
            raise RuntimeError(f"Affine successor piece {piece.label} has invalid offset shape.")

    def _piece_successor_expressions(self, piece: AffineSuccessorPiece) -> tuple[z3.ArithRef, ...]:
        expressions: list[z3.ArithRef] = []
        for row, offset in zip(piece.matrix, piece.offset):
            expr = z3.RealVal(str(float(offset)))
            for coefficient, name in zip(row, self.env.state_names):
                if float(coefficient) != 0.0:
                    expr = expr + z3.RealVal(str(float(coefficient))) * self.variables[name]
            expressions.append(expr)
        return tuple(expressions)

    def _affine_image_box_may_overlap(
        self,
        source: SymbolicBlock,
        target: SymbolicBlock,
        piece: AffineSuccessorPiece,
    ) -> bool:
        low = source.lower if np.all(source.lower <= source.upper) else self.env.bounds.low
        high = source.upper if np.all(source.lower <= source.upper) else self.env.bounds.high
        image_low: list[float] = []
        image_high: list[float] = []
        for row, offset in zip(piece.matrix, piece.offset):
            lower = float(offset)
            upper = float(offset)
            for coefficient, lo, hi in zip(row, low, high):
                coefficient = float(coefficient)
                if coefficient >= 0:
                    lower += coefficient * float(lo)
                    upper += coefficient * float(hi)
                else:
                    lower += coefficient * float(hi)
                    upper += coefficient * float(lo)
            image_low.append(lower)
            image_high.append(upper)
        return bool(
            np.all(np.asarray(image_high) >= target.lower - 1e-9)
            and np.all(np.asarray(image_low) <= target.upper + 1e-9)
        )

    def _is_sat(self, constraints: Sequence[z3.BoolRef]) -> bool:
        result = self._sat_result(constraints)
        if result == z3.unknown:
            solver = z3.Solver()
            solver.set(timeout=max(10_000, 5 * self.timeout_ms))
            solver.add(*self.env.bound_constraints(self.variables))
            solver.add(*constraints)
            result = solver.check()
        if result == z3.unknown:
            raise RuntimeError(
                "SMT satisfiability remained unknown after the extended retry; "
                "the candidate cannot be safely pruned."
            )
        return result == z3.sat

    def _sat_result_qf_lra(self, constraints: Sequence[z3.BoolRef]) -> z3.CheckSatResult:
        try:
            solver = z3.SolverFor("QF_LRA")
        except z3.Z3Exception:
            return z3.unknown
        solver.set(timeout=self.timeout_ms)
        for constraint in self.env.bound_constraints(self.variables):
            solver.add(constraint)
        for constraint in constraints:
            solver.add(constraint)
        key = "qf-lra:" + z3.And(*solver.assertions()).sexpr()
        if key not in self._sat_cache:
            self._sat_cache[key] = solver.check()
        return self._sat_cache[key]

    def _new_qf_lra_solver(self, constraints: Sequence[z3.BoolRef]) -> z3.Solver:
        try:
            solver = z3.SolverFor("QF_LRA")
        except z3.Z3Exception:
            solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        for constraint in self.env.bound_constraints(self.variables):
            solver.add(constraint)
        for constraint in constraints:
            solver.add(constraint)
        return solver

    def _sat_result(self, constraints: Sequence[z3.BoolRef]) -> z3.CheckSatResult:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        for constraint in self.env.bound_constraints(self.variables):
            solver.add(constraint)
        for constraint in constraints:
            solver.add(constraint)
        key = "general:" + z3.And(*solver.assertions()).sexpr()
        if key not in self._sat_cache:
            self._sat_cache[key] = solver.check()
        return self._sat_cache[key]

    def _witness(self, constraints: Sequence[z3.BoolRef]) -> np.ndarray | None:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        for constraint in self.env.bound_constraints(self.variables):
            solver.add(constraint)
        for constraint in constraints:
            solver.add(constraint)
        if solver.check() != z3.sat:
            return None
        model = solver.model()
        values = [
            _z3_to_float(model.eval(self.variables[name], model_completion=True))
            for name in self.env.state_names
        ]
        return self.env.bounds.clip(values)

    def _dedupe_semantic(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        buckets: dict[
            tuple[tuple[tuple[float, tuple[str, ...]], ...], tuple[tuple[str, bool], ...]],
            list[_Candidate],
        ] = {}
        for candidate in candidates:
            key = (
                tuple(sorted(candidate.action_paths.items())),
                tuple(sorted(candidate.atom_values.items())),
            )
            bucket = buckets.setdefault(key, [])
            if any(self._equivalent(candidate, old) for old in bucket):
                continue
            bucket.append(candidate)
        return [candidate for bucket in buckets.values() for candidate in bucket]

    def _equivalent(self, left: _Candidate, right: _Candidate) -> bool:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        for constraint in self.env.bound_constraints(self.variables):
            solver.add(constraint)
        solver.add(z3.Xor(left.formula(), right.formula()))
        return solver.check() == z3.unsat


def _z3_to_float(value: z3.ExprRef) -> float:
    if z3.is_rational_value(value):
        return value.numerator_as_long() / value.denominator_as_long()
    text = value.as_decimal(20).rstrip("?")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)
