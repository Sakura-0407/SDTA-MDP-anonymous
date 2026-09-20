from __future__ import annotations

from math import erf, sqrt
from typing import Any, Mapping

import numpy as np
import z3

from .actions import ContinuousActionSpec, DiscreteActionSpec
from .environment import (
    ActionPathSupportMap,
    AffineSuccessorPiece,
    ArrayLike,
    ContinuousEnvironment,
    StepResult,
    SymbolicPathCondition,
)


class StochasticActionEnvironment(ContinuousEnvironment):
    """Turn a deterministic benchmark into an action-stochastic MDP.

    Finite actions use a random-action slip model. Continuous actions use
    clipped Gaussian actuator noise. The wrapped environment still computes
    the successor cost, terminal label, and constraint status for the action
    that was actually executed.
    """

    def __init__(
        self,
        base_env: ContinuousEnvironment,
        *,
        name: str,
        slip_probability: float = 0.10,
        continuous_noise_fraction: float = 0.05,
        project_stochastic_paths: bool = False,
        gaussian_support_sigma: float = 2.0,
        gaussian_support_points: int = 3,
    ) -> None:
        if not 0.0 <= slip_probability <= 1.0:
            raise ValueError("slip_probability must lie in [0, 1]")
        if continuous_noise_fraction < 0.0:
            raise ValueError("continuous_noise_fraction must be non-negative")
        if gaussian_support_sigma <= 0.0:
            raise ValueError("gaussian_support_sigma must be positive")
        if gaussian_support_points < 2:
            raise ValueError("gaussian_support_points must be at least two")
        self.base_env = base_env
        self.name = name
        self.state_names = tuple(base_env.state_names)
        self.action_names = tuple(base_env.action_names)
        self.actions = np.asarray(base_env.actions, dtype=float).copy()
        self.bounds = base_env.bounds
        self.dt = float(base_env.dt)
        self.target_state = np.asarray(base_env.target_state, dtype=float).copy()
        self.slip_probability = float(slip_probability)
        self.continuous_noise_fraction = float(continuous_noise_fraction)
        self.project_stochastic_paths = bool(project_stochastic_paths)
        self.gaussian_support_sigma = float(gaussian_support_sigma)
        self.gaussian_support_points = int(gaussian_support_points)
        self._transition_rng = np.random.default_rng(0)
        self.reset_evaluation_action_diagnostics()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def action_spec(self) -> DiscreteActionSpec | ContinuousActionSpec:
        return self.base_env.action_spec()

    def transition_regime(self) -> str:
        return "stochastic"

    def transition_noise_scale(self) -> float:
        if self.is_continuous_action():
            return self.continuous_noise_fraction
        return self.slip_probability

    def transition_noise_model(self) -> str:
        return "gaussian-action-noise" if self.is_continuous_action() else "random-action-slip"

    def topology_noise_projection(self) -> str:
        if not self.project_stochastic_paths:
            return "none"
        if self.is_continuous_action():
            return "bounded-ksigma-representative"
        return "exact-discrete-slip"

    def topology_noise_interval_probability(self) -> float | None:
        if not self.project_stochastic_paths:
            return None
        if self.is_continuous_action():
            return float(erf(self.gaussian_support_sigma / sqrt(2.0)))
        return 1.0

    def topology_noise_support_points(self) -> int | None:
        if not self.project_stochastic_paths or not self.is_continuous_action():
            return None
        return self.gaussian_support_points

    def projected_paths_shared_across_actions(self) -> bool:
        """Whether every intended action has the same projected path formulas."""

        if not self.project_stochastic_paths:
            return False
        if not self.is_continuous_action():
            return 0.0 < self.slip_probability < 1.0
        return self.base_env.name in {
            "mountain_car_continuous",
            "pendulum_swing_up",
            "continuous_double_integrator_parking",
        }

    def base_benchmark_name(self) -> str:
        return self.base_env.name

    def set_transition_seed(self, seed: int) -> None:
        self._transition_rng = np.random.default_rng(int(seed))
        self.base_env.set_transition_seed(int(seed))

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        state = self.base_env.reset(rng)
        if rng is not None:
            transition_seed = int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))
            self._transition_rng = np.random.default_rng(transition_seed)
        return state

    def nominal_initial_state(self) -> np.ndarray:
        return self.base_env.nominal_initial_state()

    def step(self, state: ArrayLike, action: float) -> StepResult:
        if self.is_continuous_action():
            standard_noise = (float(self._transition_rng.normal()), 0.0)
        else:
            standard_noise = (
                float(self._transition_rng.random()),
                float(self._transition_rng.random()),
            )
        return self.step_with_standard_noise(state, action, standard_noise)

    def step_with_standard_noise(
        self,
        state: ArrayLike,
        action: float,
        standard_noise: tuple[float, float] | np.ndarray,
    ) -> StepResult:
        """Apply one transition using a caller-supplied standard noise draw."""
        intended_action = self.nearest_action(action)
        primary = float(standard_noise[0])
        secondary = float(standard_noise[1]) if len(standard_noise) > 1 else 0.0
        executed_action = self._executed_action_from_standard_noise(
            intended_action,
            primary,
            secondary,
        )
        return self._step_for_executed_action(
            state,
            intended_action,
            executed_action,
            primary=primary,
            secondary=secondary,
        )

    def sample_standard_noise_tape(
        self,
        rng: np.random.Generator,
        *,
        scenarios: int,
        horizon: int,
    ) -> np.ndarray:
        """Draw common random numbers reusable across candidate actions."""
        if scenarios <= 0 or horizon <= 0:
            raise ValueError("scenarios and horizon must be positive")
        if self.is_continuous_action():
            tape = np.zeros((scenarios, horizon, 2), dtype=float)
            tape[:, :, 0] = rng.normal(size=(scenarios, horizon))
            return tape
        return rng.random((scenarios, horizon, 2))

    def expected_step_outcomes(
        self,
        state: ArrayLike,
        action: float,
        *,
        quadrature_order: int = 7,
    ) -> tuple[tuple[float, StepResult], ...]:
        """Return exact finite-slip or Gauss-Hermite transition outcomes."""
        intended_action = self.nearest_action(action)
        if not self.is_continuous_action():
            alternatives = self.actions[~np.isclose(self.actions, intended_action)]
            if alternatives.size == 0 or self.slip_probability <= 0.0:
                return ((1.0, self._step_for_executed_action(state, intended_action, intended_action)),)
            outcomes: list[tuple[float, StepResult]] = [
                (
                    1.0 - self.slip_probability,
                    self._step_for_executed_action(state, intended_action, intended_action),
                )
            ]
            alternative_probability = self.slip_probability / float(alternatives.size)
            outcomes.extend(
                (
                    alternative_probability,
                    self._step_for_executed_action(
                        state,
                        intended_action,
                        float(executed_action),
                    ),
                )
                for executed_action in alternatives
            )
            return tuple(outcomes)

        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive")
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        low, high = self.action_bounds()
        sigma = self.continuous_noise_fraction * float(high[0] - low[0])
        outcomes = []
        for node, weight in zip(nodes, weights):
            executed_action = float(
                np.clip(
                    intended_action + np.sqrt(2.0) * sigma * float(node),
                    float(low[0]),
                    float(high[0]),
                )
            )
            outcomes.append(
                (
                    float(weight) / float(np.sqrt(np.pi)),
                    self._step_for_executed_action(
                        state,
                        intended_action,
                        executed_action,
                    ),
                )
            )
        return tuple(outcomes)

    def _step_for_executed_action(
        self,
        state: ArrayLike,
        intended_action: float,
        executed_action: float,
        *,
        primary: float | None = None,
        secondary: float | None = None,
    ) -> StepResult:
        result = self.base_env.step(state, executed_action)
        info = dict(result.info)
        info.update(
            {
                "transition_regime": "stochastic",
                "noise_model": self.transition_noise_model(),
                "noise_scale": self.transition_noise_scale(),
                "intended_action": float(intended_action),
                "executed_action": float(executed_action),
                "noise_draw_primary": primary,
                "noise_draw_secondary": secondary,
            }
        )
        return StepResult(
            next_state=np.asarray(result.next_state, dtype=float).copy(),
            cost=float(result.cost),
            done=bool(result.done),
            info=info,
        )

    def reset_evaluation_action_diagnostics(self) -> None:
        self._evaluation_action_samples = 0
        self._evaluation_action_perturbations = 0
        self._evaluation_abs_action_deviation = 0.0
        self._evaluation_intended_action_sum = 0.0
        self._evaluation_executed_action_sum = 0.0

    def record_evaluation_action(self, info: Mapping[str, Any]) -> None:
        intended = info.get("intended_action")
        executed = info.get("executed_action")
        if intended is None or executed is None:
            return
        intended_value = float(intended)
        executed_value = float(executed)
        deviation = abs(executed_value - intended_value)
        self._evaluation_action_samples += 1
        self._evaluation_action_perturbations += int(deviation > 1e-12)
        self._evaluation_abs_action_deviation += deviation
        self._evaluation_intended_action_sum += intended_value
        self._evaluation_executed_action_sum += executed_value

    def evaluation_action_diagnostics(self) -> dict[str, float | int | str | None]:
        samples = self._evaluation_action_samples
        if samples == 0:
            return {
                "noise_model": self.transition_noise_model(),
                "action_sample_count": 0,
                "action_perturbation_count": 0,
                "action_perturbation_rate": None,
                "mean_abs_action_deviation": None,
                "mean_intended_action": None,
                "mean_executed_action": None,
            }
        return {
            "noise_model": self.transition_noise_model(),
            "action_sample_count": samples,
            "action_perturbation_count": self._evaluation_action_perturbations,
            "action_perturbation_rate": self._evaluation_action_perturbations / samples,
            "mean_abs_action_deviation": self._evaluation_abs_action_deviation / samples,
            "mean_intended_action": self._evaluation_intended_action_sum / samples,
            "mean_executed_action": self._evaluation_executed_action_sum / samples,
        }

    def _executed_action_from_standard_noise(
        self,
        intended_action: float,
        primary: float,
        secondary: float,
    ) -> float:
        if self.is_continuous_action():
            low, high = self.action_bounds()
            sigma = self.continuous_noise_fraction * float(high[0] - low[0])
            noisy_action = float(intended_action) + sigma * primary
            return float(np.clip(noisy_action, float(low[0]), float(high[0])))

        if primary >= self.slip_probability:
            return float(intended_action)
        alternatives = self.actions[~np.isclose(self.actions, intended_action)]
        if alternatives.size == 0:
            return float(intended_action)
        index = min(int(secondary * alternatives.size), alternatives.size - 1)
        return float(alternatives[index])

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        return self.base_env.symbolic_atoms(variables)

    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        if not self.project_stochastic_paths:
            return self.base_env.symbolic_path_conditions(action)
        support_actions = self._topology_executed_action_support(float(action))
        grouped: dict[str, list[SymbolicPathCondition]] = {}
        for executed_action in support_actions:
            for path in self._base_transition_path_conditions(executed_action):
                grouped.setdefault(path.name, []).append(path)

        projected: list[SymbolicPathCondition] = []
        for name, paths in grouped.items():
            formulas = [
                z3.And(*path.constraints) if path.constraints else z3.BoolVal(True)
                for path in paths
            ]
            projected.append(
                SymbolicPathCondition(
                    name=name,
                    action=float(action),
                    constraints=(z3.simplify(z3.Or(*formulas)),),
                    atom_values={},
                    description=(
                        f"{name} is possible under {self.topology_noise_projection()} "
                        f"for intended action {float(action):g}"
                    ),
                )
            )
        return projected

    def _topology_executed_action_support(self, intended_action: float) -> tuple[float, ...]:
        intended = self.nearest_action(intended_action)
        if not self.is_continuous_action():
            if self.slip_probability <= 0.0:
                return (float(intended),)
            alternatives = tuple(
                float(action)
                for action in self.actions
                if not np.isclose(float(action), intended)
            )
            if not alternatives:
                return (float(intended),)
            # Every alternative has positive probability when slip_probability
            # is positive. The intended action belongs to the support only when
            # its remaining probability is also positive.
            if self.slip_probability < 1.0:
                return tuple(float(action) for action in self.actions)
            return alternatives

        low, high = self.action_bounds()
        sigma = self.continuous_noise_fraction * float(high[0] - low[0])
        radius = self.gaussian_support_sigma * sigma
        interval_low = max(float(low[0]), float(intended) - radius)
        interval_high = min(float(high[0]), float(intended) + radius)
        points = list(np.linspace(interval_low, interval_high, self.gaussian_support_points))
        points.append(float(intended))
        points.extend(
            float(action)
            for action in self.actions
            if interval_low - 1e-12 <= float(action) <= interval_high + 1e-12
        )
        return tuple(sorted({float(self.base_env.nearest_action(point)) for point in points}))

    def _base_transition_path_conditions(self, executed_action: float) -> list[SymbolicPathCondition]:
        if self.base_env.name in {"point_mass", "continuous_point_mass"}:
            return self._point_mass_successor_path_conditions(executed_action)
        return self.base_env.symbolic_path_conditions(executed_action)

    def _point_mass_successor_path_conditions(self, executed_action: float) -> list[SymbolicPathCondition]:
        variables = self.base_env.make_symbolic_variables()
        x, y = variables["x"], variables["y"]
        action = self.base_env.nearest_action(executed_action)
        if self.base_env.name == "point_mass":
            moves = {
                0.0: (-self.base_env.dt, 0.0),
                1.0: (self.base_env.dt, 0.0),
                2.0: (0.0, -self.base_env.dt),
                3.0: (0.0, self.base_env.dt),
                4.0: (0.0, 0.0),
            }
            dx, dy = moves[float(action)]
        else:
            dx = self.base_env.dt * float(np.cos(action))
            dy = self.base_env.dt * float(np.sin(action))
        next_x = x + z3.RealVal(str(float(dx)))
        next_y = y + z3.RealVal(str(float(dy)))
        goal = z3.And(next_x >= 0.82, next_y >= 0.82)
        obstacle = z3.And(
            next_x >= float(self.base_env.obstacle_low[0]),
            next_x <= float(self.base_env.obstacle_high[0]),
            next_y >= float(self.base_env.obstacle_low[1]),
            next_y <= float(self.base_env.obstacle_high[1]),
        )
        return [
            SymbolicPathCondition("goal", action, (goal,), {}, "declared successor goal region"),
            SymbolicPathCondition(
                "obstacle",
                action,
                (z3.And(z3.Not(goal), obstacle),),
                {},
                "declared successor obstacle region",
            ),
            SymbolicPathCondition(
                "free",
                action,
                (z3.And(z3.Not(goal), z3.Not(obstacle)),),
                {},
                "declared successor free region",
            ),
        ]

    def path_conditions_may_overlap(self) -> bool:
        return self.project_stochastic_paths or self.base_env.path_conditions_may_overlap()

    def affine_successor_pieces(
        self,
        variables: Mapping[str, Any],
        action: float,
    ) -> tuple[AffineSuccessorPiece, ...]:
        # Random execution can reach successors outside a nominal affine piece.
        return ()

    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        return self.base_env.evaluate_atoms(state)

    def atom_boundaries(self) -> dict[str, str]:
        return self.base_env.atom_boundaries()

    def safety_action(self) -> float:
        return self.base_env.safety_action()

    def unsafe_path_names(self) -> tuple[str, ...]:
        return self.base_env.unsafe_path_names()

    def symbolic_critical_actions(self) -> tuple[float, ...]:
        return self.base_env.symbolic_critical_actions()

    def state_critical_actions(self, state: ArrayLike) -> tuple[float, ...]:
        return self.base_env.state_critical_actions(state)

    def symbolic_action_set(self):
        # A projected command is not a hard safety certificate under actuator noise.
        return None

    def terminal_potential(self, state: ArrayLike) -> float:
        return self.base_env.terminal_potential(state)

    def local_policy_depth(self) -> int:
        return self.base_env.local_policy_depth()

    def local_policy_action(self, state: ArrayLike, fallback_action: float | None = None) -> float:
        # Keep the original nominal local controller; stochasticity is modeled
        # by absorption rollouts and by execution of the selected command.
        return self.base_env.local_policy_action(state, fallback_action)

    def block_role(
        self,
        witness: ArrayLike,
        atom_values: Mapping[str, bool],
        action_paths: ActionPathSupportMap,
    ) -> str:
        return self.base_env.block_role(witness, atom_values, action_paths)

    def bounds_from_atom_assignment(
        self,
        atom_values: Mapping[str, bool],
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base_env.bounds_from_atom_assignment(atom_values, eps)

    def refine_bounds_from_paths(
        self,
        action_paths: ActionPathSupportMap,
        low: np.ndarray,
        high: np.ndarray,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base_env.refine_bounds_from_paths(action_paths, low, high, eps)
