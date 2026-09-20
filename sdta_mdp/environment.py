from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .actions import ContinuousActionSpec, DiscreteActionSpec
from .exact_actions import SymbolicActionSet

try:
    import z3
except ImportError:  # pragma: no cover - pyproject 默认依赖 z3-solver
    z3 = None


ArrayLike = Sequence[float] | np.ndarray
PathSupport = tuple[str, ...]
ActionPathSupportMap = Mapping[float, PathSupport]


@dataclass(frozen=True)
class StepResult:
    """环境一步执行结果。"""

    next_state: np.ndarray
    cost: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class StateBounds:
    """连续状态空间的盒约束。"""

    low: np.ndarray
    high: np.ndarray

    def clip(self, state: ArrayLike) -> np.ndarray:
        return np.clip(np.asarray(state, dtype=float), self.low, self.high)


@dataclass(frozen=True)
class SymbolicPathCondition:
    """单步白盒分支对应的路径条件。"""

    name: str
    action: float
    constraints: tuple[Any, ...]
    atom_values: Mapping[str, bool]
    description: str


@dataclass(frozen=True)
class AffineSuccessorPiece:
    """One exact-LRA successor branch x' = matrix x + offset under guard."""

    guard: tuple[Any, ...]
    matrix: np.ndarray
    offset: np.ndarray
    label: str


class ContinuousEnvironment(ABC):
    """连续状态 MDP 的白盒接口。

    子类同时提供数值执行和符号路径条件。符号路径条件是 SDTA-MDP 自动
    分区的入口，用来替代人工 Foster-Lyapunov shell 或手工网格分区。
    """

    name: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    actions: np.ndarray
    bounds: StateBounds
    dt: float
    target_state: np.ndarray

    def action_spec(self) -> DiscreteActionSpec | ContinuousActionSpec:
        """返回环境动作空间规格；旧环境默认使用有限动作。"""

        return DiscreteActionSpec(np.asarray(self.actions, dtype=float))

    def is_continuous_action(self) -> bool:
        return self.action_spec().continuous

    def transition_regime(self) -> str:
        return "deterministic"

    def transition_noise_scale(self) -> float:
        return 0.0

    def set_transition_seed(self, seed: int) -> None:
        return None

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        spec = self.action_spec()
        if isinstance(spec, ContinuousActionSpec):
            return spec.low.copy(), spec.high.copy()
        values = np.asarray(spec.actions, dtype=float)
        return np.asarray([float(values.min())]), np.asarray([float(values.max())])

    def action_candidates(self, fallback_action: float | None = None) -> np.ndarray:
        """返回全局或局部规划使用的候选动作。"""

        spec = self.action_spec()
        if isinstance(spec, ContinuousActionSpec):
            if fallback_action is None:
                return spec.initial_candidates()
            return spec.refine_candidates(float(fallback_action))
        return np.asarray(spec.actions, dtype=float)

    def symbolic_critical_actions(self) -> tuple[float, ...]:
        """连续动作候选集中应保留的环境临界动作。"""

        spec = self.action_spec()
        return spec.critical_values if isinstance(spec, ContinuousActionSpec) else ()

    def state_critical_actions(self, state: ArrayLike) -> tuple[float, ...]:
        """返回依赖当前状态的解析临界动作；默认没有额外候选。"""

        return ()

    def local_action_candidates(self, state: ArrayLike, fallback_action: float | None = None) -> np.ndarray:
        """合并连续动作网格和状态相关临界动作，供 SDTA 局部策略改进使用。"""

        candidates = np.concatenate(
            [
                self.action_candidates(fallback_action),
                np.asarray(self.state_critical_actions(state), dtype=float),
            ]
        )
        return np.unique(np.asarray([self.nearest_action(action) for action in candidates], dtype=float))

    def symbolic_action_set(self) -> SymbolicActionSet | None:
        """返回完全符号化的安全动作集合；复杂非线性环境可以不实现。"""

        return None

    def supports_exact_action_projection(self) -> bool:
        return self.symbolic_action_set() is not None

    def exact_safe_action_interval(self, state: ArrayLike):
        action_set = self.symbolic_action_set()
        if action_set is None:
            return None
        return action_set.interval_for_state(self.state_names, self.bounds.clip(state))

    def exact_policy_action(self, state: ArrayLike, fallback_action: float | None = None) -> float:
        """在完整安全动作区间内做局部连续优化。"""

        interval = self.exact_safe_action_interval(state)
        if interval is None or not interval.feasible:
            return self.adaptive_policy_action(state, fallback_action)
        state_arr = self.bounds.clip(state)
        midpoint = 0.5 * (interval.low + interval.high)
        candidates = np.asarray([interval.low, midpoint, interval.high], dtype=float)
        return float(min(candidates, key=lambda action: self._lookahead_score(state_arr, float(action), self.local_policy_depth())))

    @abstractmethod
    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """采样一个初始状态。"""

    @abstractmethod
    def nominal_initial_state(self) -> np.ndarray:
        """返回代表性初始状态。"""

    @abstractmethod
    def step(self, state: ArrayLike, action: float) -> StepResult:
        """执行一步环境转移。"""

    @abstractmethod
    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        """返回用于诱导分区的原子谓词。"""

    @abstractmethod
    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        """返回给定动作下所有白盒分支路径条件。"""

    def path_conditions_may_overlap(self) -> bool:
        """Whether projected path conditions for one action may overlap."""

        return False

    def affine_successor_pieces(self, variables: Mapping[str, Any], action: float) -> tuple[AffineSuccessorPiece, ...]:
        """Return exact-LRA affine successor pieces for a representative action."""

        return ()

    @abstractmethod
    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        """在具体状态上计算原子谓词真假。"""

    @abstractmethod
    def atom_boundaries(self) -> dict[str, str]:
        """返回原子谓词对应的边界说明。"""

    @abstractmethod
    def safety_action(self) -> float:
        """远端/边缘块使用的固定安全策略。"""

    def unsafe_path_names(self) -> tuple[str, ...]:
        """Return declared path labels that denote an unsafe successor."""

        return ()

    def block_role(
        self,
        witness: ArrayLike,
        atom_values: Mapping[str, bool],
        action_paths: ActionPathSupportMap,
    ) -> str:
        """Return the environment-supplied frontier role for a symbolic block."""

        atoms = dict(atom_values) if atom_values else self.evaluate_atoms(witness)
        paths = {
            path
            for support in action_paths.values()
            for path in support
        }
        if atoms.get("goal_reached", False) and not atoms.get("stopped", False):
            return "edge"
        if paths and all(path in {"already_collision"} for path in paths):
            return "edge"
        return "core"

    def make_symbolic_variables(self) -> dict[str, Any]:
        if z3 is None:
            raise RuntimeError("需要 z3-solver 才能构造符号变量。")
        return {name: z3.Real(name) for name in self.state_names}

    def bound_constraints(self, variables: Mapping[str, Any]) -> list[Any]:
        constraints: list[Any] = []
        for idx, name in enumerate(self.state_names):
            constraints.append(variables[name] >= float(self.bounds.low[idx]))
            constraints.append(variables[name] <= float(self.bounds.high[idx]))
        return constraints

    def action_index(self, action: float) -> int:
        return int(np.argmin(np.abs(self.actions - float(action))))

    def nearest_action(self, action: float) -> float:
        spec = self.action_spec()
        if isinstance(spec, ContinuousActionSpec):
            return spec.clip_scalar(action)
        return float(self.actions[self.action_index(action)])

    def distance_to_target(self, state: ArrayLike) -> float:
        state_arr = self.bounds.clip(state)
        return float(np.linalg.norm(state_arr - self.target_state))

    def terminal_potential(self, state: ArrayLike) -> float:
        """局部连续策略改进使用的终端势函数，越小越好。"""

        return self.distance_to_target(state)

    def local_policy_depth(self) -> int:
        """默认局部 lookahead 深度。"""

        return 2

    def local_policy_action(self, state: ArrayLike, fallback_action: float | None = None) -> float:
        """基于白盒动力学的局部连续状态反馈动作。

        这是 DTPTA 第二阶段的工程化实现：frontier SMDP 给出全局结构，
        具体连续状态上再做短视滚动优化，避免“一个符号块一个常数动作”
        过粗造成的控制性能损失。
        """

        return self.adaptive_policy_action(state, fallback_action)

    def adaptive_policy_action(self, state: ArrayLike, fallback_action: float | None = None) -> float:
        """使用自适应候选集执行局部 lookahead。"""

        state_arr = self.bounds.clip(state)
        candidates = self.local_action_candidates(state_arr, fallback_action)
        best_action = float(candidates[0])
        best_score = float("inf")
        for action in candidates:
            score = self._lookahead_score(state_arr, float(action), self.local_policy_depth())
            if score < best_score:
                best_score = score
                best_action = float(action)
        return best_action

    def _lookahead_score(self, state: np.ndarray, action: float, depth: int) -> float:
        step = self.step(state, action)
        penalty = 100.0 if step.info.get("constraint_violation", False) else 0.0
        if depth <= 1 or step.done:
            return step.cost + penalty + 5.0 * self.terminal_potential(step.next_state)
        return step.cost + penalty + min(
            self._lookahead_score(step.next_state, float(next_action), depth - 1)
            for next_action in self.local_action_candidates(step.next_state, action)
        )

    def bounds_from_atom_assignment(
        self,
        atom_values: Mapping[str, bool],
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把谓词赋值转成一个保守盒区域，用于快速抽样和状态归属。"""

        return self.bounds.low.copy(), self.bounds.high.copy()

    def refine_bounds_from_paths(
        self,
        action_paths: ActionPathSupportMap,
        low: np.ndarray,
        high: np.ndarray,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """用路径签名进一步收紧数值盒约束；默认不收紧。"""

        return low, high


def _z3_real(value: float) -> Any:
    if z3 is None:
        raise RuntimeError("z3-solver is required for symbolic successor expressions.")
    return z3.RealVal(str(float(value)))


def _affine_clip_pieces(
    variables: Mapping[str, Any],
    state_names: Sequence[str],
    matrix: np.ndarray,
    offset: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str,
) -> tuple[AffineSuccessorPiece, ...]:
    matrix = np.asarray(matrix, dtype=float)
    offset = np.asarray(offset, dtype=float)
    dim = len(state_names)
    axis_cases: list[list[tuple[tuple[Any, ...], np.ndarray, float, str]]] = []
    for axis in range(dim):
        row = matrix[axis].astype(float)
        raw = _linear_expr(variables, state_names, row, float(offset[axis]))
        zero = np.zeros(dim, dtype=float)
        axis_cases.append(
            [
                ((raw <= _z3_real(float(low[axis])),), zero, float(low[axis]), f"{state_names[axis]}=low"),
                (
                    (raw >= _z3_real(float(low[axis])), raw <= _z3_real(float(high[axis]))),
                    row,
                    float(offset[axis]),
                    f"{state_names[axis]}=affine",
                ),
                ((raw >= _z3_real(float(high[axis])),), zero, float(high[axis]), f"{state_names[axis]}=high"),
            ]
        )

    pieces: list[AffineSuccessorPiece] = []
    for combo in product(*axis_cases):
        guards: list[Any] = []
        rows: list[np.ndarray] = []
        offsets: list[float] = []
        labels: list[str] = []
        for guard, row, axis_offset, axis_label in combo:
            guards.extend(guard)
            rows.append(row)
            offsets.append(axis_offset)
            labels.append(axis_label)
        pieces.append(
            AffineSuccessorPiece(
                guard=tuple(guards),
                matrix=np.asarray(rows, dtype=float),
                offset=np.asarray(offsets, dtype=float),
                label=f"{label}:{','.join(labels)}",
            )
        )
    return tuple(pieces)


def _linear_expr(
    variables: Mapping[str, Any],
    state_names: Sequence[str],
    row: np.ndarray,
    offset: float,
) -> Any:
    expr = _z3_real(float(offset))
    for coefficient, name in zip(row, state_names):
        if float(coefficient) != 0.0:
            expr = expr + _z3_real(float(coefficient)) * variables[name]
    return expr


class BrakingCarEnvironment(ContinuousEnvironment):
    """来自 SymPar braking-car artifact 的连续刹车车 MDP。"""

    def __init__(self) -> None:
        self.name = "braking_car"
        self.state_names = ("velocity", "position")
        self.action_names = ("急刹", "强刹", "中刹", "轻刹", "微刹", "极微刹", "滑行制动")
        self.actions = np.asarray([-10.0, -5.0, -2.5, -0.5, -0.05, -0.01, -0.001], dtype=float)
        self.bounds = StateBounds(np.asarray([0.0, 0.0]), np.asarray([15.0, 20.0]))
        self.dt = 2.0
        self.goal_position = 10.0
        self.mid_position = 5.0
        self.target_state = np.asarray([0.0, 9.5])
        self.eps = 1e-9

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([rng.uniform(0.5, 10.0), rng.uniform(0.0, 8.5)])

    def nominal_initial_state(self) -> np.ndarray:
        return np.asarray([7.0, 1.0])

    def safety_action(self) -> float:
        return float(np.min(self.actions))

    def unsafe_path_names(self) -> tuple[str, ...]:
        return ("already_collision",)

    def terminal_potential(self, state: ArrayLike) -> float:
        velocity, position = self.bounds.clip(state)
        if position >= self.goal_position and velocity > self.eps:
            return 100.0 + 10.0 * float(velocity)
        return abs(float(velocity)) + 0.15 * max(0.0, self.goal_position - float(position))

    def step(self, state: ArrayLike, action: float) -> StepResult:
        velocity, position = self.bounds.clip(state)
        action = self.nearest_action(action)
        if velocity <= self.eps:
            return StepResult(np.asarray([0.0, position]), 0.0, True, {"terminal": "stopped"})
        if position >= self.goal_position:
            return StepResult(np.asarray([velocity, self.goal_position]), 100.0, True, {"terminal": "collision"})

        stop_time = -float(velocity) / action
        transition_time = min(stop_time, self.dt)
        raw_position = position + velocity * transition_time + 0.5 * action * transition_time**2
        next_position = min(float(raw_position), self.goal_position)
        next_velocity = 0.0 if next_position >= self.goal_position else max(float(velocity + action * transition_time), 0.0)
        collision = next_position >= self.goal_position
        stopped = next_velocity <= self.eps
        cost = 100.0 if collision else -action + (0.0 if stopped else 0.05)
        return StepResult(
            np.asarray([next_velocity, next_position]),
            float(cost),
            bool(collision or stopped),
            {"terminal": "collision" if collision else ("stopped" if stopped else None), "transition_time": transition_time},
        )

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        v, p = variables["velocity"], variables["position"]
        return {
            "stopped": v <= 0,
            "position_lt_5": p < self.mid_position,
            "position_lt_10": p < self.goal_position,
            "goal_reached": p >= self.goal_position,
        }

    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        variables = self.make_symbolic_variables()
        atoms = self.symbolic_atoms(variables)
        action = self.nearest_action(action)
        threshold = z3.RealVal(str(-action * self.dt))
        v = variables["velocity"]
        stop_in_window = v <= threshold
        rows = [
            ("already_stopped", {"stopped": True}, [atoms["stopped"]], "速度已为零。"),
            ("already_collision", {"stopped": False, "goal_reached": True}, [z3.Not(atoms["stopped"]), atoms["goal_reached"]], "已经越过目标线。"),
            ("far_stop_in_window", {"stopped": False, "position_lt_5": True, "position_lt_10": True}, [z3.Not(atoms["stopped"]), atoms["position_lt_5"], atoms["position_lt_10"], stop_in_window], "远端且本动作可停住。"),
            ("far_continue", {"stopped": False, "position_lt_5": True, "position_lt_10": True}, [z3.Not(atoms["stopped"]), atoms["position_lt_5"], atoms["position_lt_10"], z3.Not(stop_in_window)], "远端且继续运动。"),
            ("near_stop_in_window", {"stopped": False, "position_lt_5": False, "position_lt_10": True}, [z3.Not(atoms["stopped"]), z3.Not(atoms["position_lt_5"]), atoms["position_lt_10"], stop_in_window], "近端且本动作可停住。"),
            ("near_continue", {"stopped": False, "position_lt_5": False, "position_lt_10": True}, [z3.Not(atoms["stopped"]), z3.Not(atoms["position_lt_5"]), atoms["position_lt_10"], z3.Not(stop_in_window)], "近端且继续运动。"),
        ]
        return [SymbolicPathCondition(name, action, tuple(cons), values, desc) for name, values, cons, desc in rows]

    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        v, p = self.bounds.clip(state)
        return {
            "stopped": bool(v <= self.eps),
            "position_lt_5": bool(p < self.mid_position),
            "position_lt_10": bool(p < self.goal_position),
            "goal_reached": bool(p >= self.goal_position),
        }

    def atom_boundaries(self) -> dict[str, str]:
        return {
            "stopped": "velocity = 0",
            "position_lt_5": "position = 5",
            "position_lt_10": "position = 10",
            "goal_reached": "position = 10",
        }

    def block_role(
        self,
        witness: ArrayLike,
        atom_values: Mapping[str, bool],
        action_paths: ActionPathSupportMap,
    ) -> str:
        role = super().block_role(witness, atom_values, action_paths)
        if role != "core":
            return role
        velocity, position = self.bounds.clip(witness)
        if position >= self.mid_position and velocity > 10.0:
            return "remote"
        return "core"

    def bounds_from_atom_assignment(self, atom_values: Mapping[str, bool], eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.bounds.low.copy(), self.bounds.high.copy()
        if atom_values.get("stopped") is True:
            high[0] = min(high[0], 0.0)
        elif atom_values.get("stopped") is False:
            low[0] = max(low[0], eps)
        if atom_values.get("position_lt_5") is True:
            high[1] = min(high[1], self.mid_position - eps)
        elif atom_values.get("position_lt_5") is False:
            low[1] = max(low[1], self.mid_position)
        if atom_values.get("position_lt_10") is True:
            high[1] = min(high[1], self.goal_position - eps)
        elif atom_values.get("position_lt_10") is False:
            low[1] = max(low[1], self.goal_position)
        if atom_values.get("goal_reached") is True:
            low[1] = max(low[1], self.goal_position)
        elif atom_values.get("goal_reached") is False:
            high[1] = min(high[1], self.goal_position - eps)
        return low, high

    def refine_bounds_from_paths(self, action_paths: ActionPathSupportMap, low: np.ndarray, high: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        for action, support in action_paths.items():
            if len(support) != 1:
                continue
            path = support[0]
            threshold = -float(action) * self.dt
            if path.endswith("stop_in_window"):
                high[0] = min(high[0], threshold)
            elif path.endswith("continue"):
                low[0] = max(low[0], threshold + eps)
        return low, high




class PointMassNavigationEnvironment(ContinuousEnvironment):
    """二维连续导航任务：点质量需要到达右上角目标，同时避开中心危险区。"""

    def __init__(self) -> None:
        self.name = "point_mass"
        self.state_names = ("x", "y")
        self.action_names = ("左", "右", "下", "上", "停")
        self.actions = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        self.bounds = StateBounds(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]))
        self.dt = 0.08
        self.target_state = np.asarray([0.9, 0.9])
        self.obstacle_low = np.asarray([0.42, 0.42])
        self.obstacle_high = np.asarray([0.62, 0.62])

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2)])

    def nominal_initial_state(self) -> np.ndarray:
        return np.asarray([0.1, 0.1])

    def safety_action(self) -> float:
        return 4.0

    def unsafe_path_names(self) -> tuple[str, ...]:
        return ("obstacle",)

    def local_policy_depth(self) -> int:
        return 3

    def step(self, state: ArrayLike, action: float) -> StepResult:
        state_arr = self.bounds.clip(state)
        moves = {
            0.0: np.asarray([-self.dt, 0.0]),
            1.0: np.asarray([self.dt, 0.0]),
            2.0: np.asarray([0.0, -self.dt]),
            3.0: np.asarray([0.0, self.dt]),
            4.0: np.asarray([0.0, 0.0]),
        }
        next_state = self.bounds.clip(state_arr + moves[self.nearest_action(action)])
        in_obstacle = bool(np.all(next_state >= self.obstacle_low) and np.all(next_state <= self.obstacle_high))
        at_goal = self.distance_to_target(next_state) <= 0.08
        cost = self.distance_to_target(next_state) + 0.02
        if in_obstacle:
            cost += 20.0
        return StepResult(next_state, float(cost), bool(at_goal), {"terminal": "goal" if at_goal else None, "constraint_violation": in_obstacle, "transition_time": 1.0})

    def affine_successor_pieces(self, variables: Mapping[str, Any], action: float) -> tuple[AffineSuccessorPiece, ...]:
        action = self.nearest_action(action)
        moves = {
            0.0: (-self.dt, 0.0),
            1.0: (self.dt, 0.0),
            2.0: (0.0, -self.dt),
            3.0: (0.0, self.dt),
            4.0: (0.0, 0.0),
        }
        dx, dy = moves[action]
        return _affine_clip_pieces(
            variables,
            self.state_names,
            np.eye(2),
            np.asarray([dx, dy], dtype=float),
            self.bounds.low,
            self.bounds.high,
            label=f"point-mass-a{action:g}",
        )

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        x, y = variables["x"], variables["y"]
        return {
            "left": x < 0.4,
            "right": x > 0.65,
            "bottom": y < 0.4,
            "top": y > 0.65,
            "goal_zone": z3.And(x >= 0.82, y >= 0.82),
            "obstacle_x": z3.And(x >= 0.42, x <= 0.62),
            "obstacle_y": z3.And(y >= 0.42, y <= 0.62),
        }

    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        atoms = self.symbolic_atoms(self.make_symbolic_variables())
        action = self.nearest_action(action)
        not_obstacle = z3.Not(z3.And(atoms["obstacle_x"], atoms["obstacle_y"]))
        rows = [
            ("goal", {"goal_zone": True}, [atoms["goal_zone"]], "目标区域。"),
            ("obstacle", {"goal_zone": False, "obstacle_x": True, "obstacle_y": True}, [z3.Not(atoms["goal_zone"]), atoms["obstacle_x"], atoms["obstacle_y"]], "障碍区域。"),
            ("south_west", {"goal_zone": False, "left": True, "bottom": True}, [z3.Not(atoms["goal_zone"]), not_obstacle, atoms["left"], atoms["bottom"]], "左下区域。"),
            ("north_west", {"goal_zone": False, "left": True, "bottom": False}, [z3.Not(atoms["goal_zone"]), not_obstacle, atoms["left"], z3.Not(atoms["bottom"])], "左上区域。"),
            ("south_east", {"goal_zone": False, "left": False, "bottom": True}, [z3.Not(atoms["goal_zone"]), not_obstacle, z3.Not(atoms["left"]), atoms["bottom"]], "右下区域。"),
            ("north_east", {"goal_zone": False, "left": False, "bottom": False}, [z3.Not(atoms["goal_zone"]), not_obstacle, z3.Not(atoms["left"]), z3.Not(atoms["bottom"])], "右上区域。"),
        ]
        return [SymbolicPathCondition(name, action, tuple(cons), values, desc) for name, values, cons, desc in rows]

    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        x, y = self.bounds.clip(state)
        return {
            "left": bool(x < 0.4),
            "right": bool(x > 0.65),
            "bottom": bool(y < 0.4),
            "top": bool(y > 0.65),
            "goal_zone": bool(x >= 0.82 and y >= 0.82),
            "obstacle_x": bool(0.42 <= x <= 0.62),
            "obstacle_y": bool(0.42 <= y <= 0.62),
        }

    def atom_boundaries(self) -> dict[str, str]:
        return {"left": "x = 0.4", "right": "x = 0.65", "bottom": "y = 0.4", "top": "y = 0.65", "goal_zone": "x = 0.82 or y = 0.82", "obstacle_x": "x = 0.42/0.62", "obstacle_y": "y = 0.42/0.62"}

    def bounds_from_atom_assignment(self, atom_values: Mapping[str, bool], eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.bounds.low.copy(), self.bounds.high.copy()
        if atom_values.get("left") is True:
            high[0] = min(high[0], 0.4 - eps)
        elif atom_values.get("left") is False:
            low[0] = max(low[0], 0.4)
        if atom_values.get("right") is True:
            low[0] = max(low[0], 0.65 + eps)
        elif atom_values.get("right") is False:
            high[0] = min(high[0], 0.65)
        if atom_values.get("bottom") is True:
            high[1] = min(high[1], 0.4 - eps)
        elif atom_values.get("bottom") is False:
            low[1] = max(low[1], 0.4)
        if atom_values.get("top") is True:
            low[1] = max(low[1], 0.65 + eps)
        elif atom_values.get("top") is False:
            high[1] = min(high[1], 0.65)
        if atom_values.get("goal_zone") is True:
            low[0] = max(low[0], 0.82)
            low[1] = max(low[1], 0.82)
        if atom_values.get("obstacle_x") is True:
            low[0] = max(low[0], self.obstacle_low[0])
            high[0] = min(high[0], self.obstacle_high[0])
        if atom_values.get("obstacle_y") is True:
            low[1] = max(low[1], self.obstacle_low[1])
            high[1] = min(high[1], self.obstacle_high[1])
        return low, high






class DoubleIntegratorParkingEnvironment(ContinuousEnvironment):
    """一维停车/双积分器任务：用离散加速度把车停到原点附近。"""

    def __init__(self) -> None:
        self.name = "double_integrator_parking"
        self.state_names = ("position", "velocity")
        self.action_names = ("左加速", "保持", "右加速")
        self.actions = np.asarray([-1.0, 0.0, 1.0])
        self.bounds = StateBounds(np.asarray([-2.0, -2.0]), np.asarray([2.0, 2.0]))
        self.dt = 0.2
        self.target_state = np.asarray([0.0, 0.0])

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([rng.uniform(-1.5, 1.5), rng.uniform(-0.6, 0.6)])

    def nominal_initial_state(self) -> np.ndarray:
        return np.asarray([1.2, 0.0])

    def safety_action(self) -> float:
        return 0.0

    def local_policy_depth(self) -> int:
        return 4

    def terminal_potential(self, state: ArrayLike) -> float:
        position, velocity = self.bounds.clip(state)
        return abs(float(position)) + 0.6 * abs(float(velocity))

    def step(self, state: ArrayLike, action: float) -> StepResult:
        p, v = self.bounds.clip(state)
        a = self.nearest_action(action)
        next_v = float(np.clip(v + self.dt * a, self.bounds.low[1], self.bounds.high[1]))
        next_p = float(np.clip(p + self.dt * v + 0.5 * self.dt**2 * a, self.bounds.low[0], self.bounds.high[0]))
        next_state = np.asarray([next_p, next_v])
        done = abs(next_p) <= 0.08 and abs(next_v) <= 0.08
        cost = next_p**2 + 0.2 * next_v**2 + 0.05 * a**2
        return StepResult(next_state, float(cost), bool(done), {"terminal": "parked" if done else None, "transition_time": 1.0})

    def affine_successor_pieces(self, variables: Mapping[str, Any], action: float) -> tuple[AffineSuccessorPiece, ...]:
        a = self.nearest_action(action)
        matrix = np.asarray([[1.0, self.dt], [0.0, 1.0]], dtype=float)
        offset = np.asarray([0.5 * self.dt**2 * a, self.dt * a], dtype=float)
        return _affine_clip_pieces(
            variables,
            self.state_names,
            matrix,
            offset,
            self.bounds.low,
            self.bounds.high,
            label=f"double-integrator-a{a:g}",
        )

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        p, v = variables["position"], variables["velocity"]
        return {
            "left": p < -0.25,
            "right": p > 0.25,
            "moving_left": v < -0.1,
            "moving_right": v > 0.1,
            "parked": z3.And(p >= -0.08, p <= 0.08, v >= -0.08, v <= 0.08),
        }

    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        atoms = self.symbolic_atoms(self.make_symbolic_variables())
        action = self.nearest_action(action)
        rows = [
            ("parked", {"parked": True}, [atoms["parked"]], "已经停稳。"),
            ("left_fast", {"parked": False, "left": True, "moving_left": True}, [z3.Not(atoms["parked"]), atoms["left"], atoms["moving_left"]], "左侧且继续向左。"),
            ("left_return", {"parked": False, "left": True, "moving_left": False}, [z3.Not(atoms["parked"]), atoms["left"], z3.Not(atoms["moving_left"])], "左侧但未向左远离。"),
            ("right_fast", {"parked": False, "right": True, "moving_right": True}, [z3.Not(atoms["parked"]), atoms["right"], atoms["moving_right"]], "右侧且继续向右。"),
            ("right_return", {"parked": False, "right": True, "moving_right": False}, [z3.Not(atoms["parked"]), atoms["right"], z3.Not(atoms["moving_right"])], "右侧但未向右远离。"),
            ("center", {"parked": False, "left": False, "right": False}, [z3.Not(atoms["parked"]), z3.Not(atoms["left"]), z3.Not(atoms["right"])], "中心区域。"),
        ]
        return [SymbolicPathCondition(name, action, tuple(cons), values, desc) for name, values, cons, desc in rows]

    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        p, v = self.bounds.clip(state)
        return {"left": bool(p < -0.25), "right": bool(p > 0.25), "moving_left": bool(v < -0.1), "moving_right": bool(v > 0.1), "parked": bool(abs(p) <= 0.08 and abs(v) <= 0.08)}

    def atom_boundaries(self) -> dict[str, str]:
        return {"left": "position = -0.25", "right": "position = 0.25", "moving_left": "velocity = -0.1", "moving_right": "velocity = 0.1", "parked": "|position| = 0.08, |velocity| = 0.08"}

    def bounds_from_atom_assignment(self, atom_values: Mapping[str, bool], eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.bounds.low.copy(), self.bounds.high.copy()
        if atom_values.get("parked") is True:
            low[0] = max(low[0], -0.08)
            high[0] = min(high[0], 0.08)
            low[1] = max(low[1], -0.08)
            high[1] = min(high[1], 0.08)
            return low, high
        if atom_values.get("left") is True:
            high[0] = min(high[0], -0.25 - eps)
        elif atom_values.get("left") is False:
            low[0] = max(low[0], -0.25)
        if atom_values.get("right") is True:
            low[0] = max(low[0], 0.25 + eps)
        elif atom_values.get("right") is False:
            high[0] = min(high[0], 0.25)
        if atom_values.get("moving_left") is True:
            high[1] = min(high[1], -0.1 - eps)
        elif atom_values.get("moving_left") is False:
            low[1] = max(low[1], -0.1)
        if atom_values.get("moving_right") is True:
            low[1] = max(low[1], 0.1 + eps)
        elif atom_values.get("moving_right") is False:
            high[1] = min(high[1], 0.1)
        return low, high
