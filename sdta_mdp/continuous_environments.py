from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import z3

from .actions import ContinuousActionSpec
from .exact_actions import SymbolicActionSet
from .environment import AffineSuccessorPiece, ArrayLike, BrakingCarEnvironment, ContinuousEnvironment, DoubleIntegratorParkingEnvironment, StateBounds, StepResult, SymbolicPathCondition, _affine_clip_pieces


class ContinuousBrakingCarEnvironment(BrakingCarEnvironment):
    """连续制动力版本的 braking car。"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "continuous_braking_car"
        self.actions = self.action_spec().initial_candidates()

    def action_spec(self) -> ContinuousActionSpec:
        return ContinuousActionSpec(
            low=np.asarray([-10.0]),
            high=np.asarray([-0.001]),
            shape=(1,),
            critical_values=(-5.0, -2.5, -0.5, -0.05, -0.01),
            max_candidates=13,
        )

    def symbolic_action_set(self) -> SymbolicActionSet:
        """精确表示一个时间窗内可停车的制动力集合。"""

        variables = self.make_symbolic_variables()
        action = z3.Real("braking_action")
        return SymbolicActionSet(
            name="braking_safe_stop",
            state_variables=variables,
            action_variable=action,
            action_low=-10.0,
            action_high=-0.001,
            safe_formula=variables["velocity"] <= -self.dt * action,
            description="制动力使车辆在一个时间窗内停住。",
        )




class ContinuousDoubleIntegratorParkingEnvironment(DoubleIntegratorParkingEnvironment):
    """连续加速度双积分器停车任务。"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "continuous_double_integrator_parking"
        self.actions = self.action_spec().initial_candidates()

    def action_spec(self) -> ContinuousActionSpec:
        return ContinuousActionSpec(
            low=np.asarray([-1.0]),
            high=np.asarray([1.0]),
            shape=(1,),
            critical_values=(-0.5, 0.5),
            max_candidates=9,
        )

    def symbolic_action_set(self) -> SymbolicActionSet:
        """精确表示下一步不会扩大位置绝对值的加速度集合。"""

        variables = self.make_symbolic_variables()
        action = z3.Real("parking_action")
        position, velocity = variables["position"], variables["velocity"]
        next_position = position + self.dt * velocity + 0.5 * self.dt**2 * action
        non_expansive = z3.If(position >= 0, z3.And(next_position >= 0, next_position <= position), z3.And(next_position <= 0, next_position >= position))
        return SymbolicActionSet(
            name="parking_non_expansive_position",
            state_variables=variables,
            action_variable=action,
            action_low=-1.0,
            action_high=1.0,
            safe_formula=non_expansive,
            description="下一步位置保持同侧且不扩大到原点的距离。",
        )


class ContinuousPointMassNavigationEnvironment(ContinuousEnvironment):
    """连续航向角二维导航：动作是 [-pi, pi] 内的航向角。"""

    def __init__(self) -> None:
        self.name = "continuous_point_mass"
        self.state_names = ("x", "y")
        self.action_names = ("航向角",)
        self.bounds = StateBounds(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]))
        self.dt = 0.08
        self.target_state = np.asarray([0.9, 0.9])
        self.obstacle_low = np.asarray([0.42, 0.42])
        self.obstacle_high = np.asarray([0.62, 0.62])
        self.actions = self.action_spec().initial_candidates()

    def action_spec(self) -> ContinuousActionSpec:
        return ContinuousActionSpec(
            low=np.asarray([-np.pi]),
            high=np.asarray([np.pi]),
            shape=(1,),
            critical_values=(-np.pi / 2, np.pi / 4, np.pi / 2),
            max_candidates=11,
        )

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2)])

    def nominal_initial_state(self) -> np.ndarray:
        return np.asarray([0.1, 0.1])

    def safety_action(self) -> float:
        return float(np.pi / 2)

    def unsafe_path_names(self) -> tuple[str, ...]:
        return ("obstacle",)

    def local_policy_depth(self) -> int:
        # 状态相关几何候选已编码绕障方向，一步局部改进即可避免组合展开。
        return 1

    def state_critical_actions(self, state: ArrayLike) -> tuple[float, ...]:
        """用目标方向和障碍绕行航点补充状态相关候选航向。"""

        state_arr = self.bounds.clip(state)
        waypoints = (
            self.target_state,
            np.asarray([0.40, 0.70]),
            np.asarray([0.70, 0.40]),
            np.asarray([0.40, 0.66]),
            np.asarray([0.66, 0.40]),
        )
        return tuple(float(np.arctan2(point[1] - state_arr[1], point[0] - state_arr[0])) for point in waypoints)

    def step(self, state: ArrayLike, action: float) -> StepResult:
        state_arr = self.bounds.clip(state)
        heading = self.nearest_action(action)
        move = self.dt * np.asarray([np.cos(heading), np.sin(heading)])
        next_state = self.bounds.clip(state_arr + move)
        in_obstacle = bool(np.all(next_state >= self.obstacle_low) and np.all(next_state <= self.obstacle_high))
        at_goal = self.distance_to_target(next_state) <= 0.08
        cost = self.distance_to_target(next_state) + 0.01 * heading**2
        if in_obstacle:
            cost += 20.0
        return StepResult(next_state, float(cost), bool(at_goal), {"terminal": "goal" if at_goal else None, "constraint_violation": in_obstacle, "transition_time": 1.0})

    def affine_successor_pieces(self, variables: Mapping[str, Any], action: float) -> tuple[AffineSuccessorPiece, ...]:
        heading = self.nearest_action(action)
        dx = self.dt * float(np.cos(heading))
        dy = self.dt * float(np.sin(heading))
        return _affine_clip_pieces(
            variables,
            self.state_names,
            np.eye(2),
            np.asarray([dx, dy], dtype=float),
            self.bounds.low,
            self.bounds.high,
            label=f"continuous-point-mass-a{heading:.6g}",
        )

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        x, y = variables["x"], variables["y"]
        return {
            "left": x < 0.4,
            "bottom": y < 0.4,
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
            "bottom": bool(y < 0.4),
            "goal_zone": bool(x >= 0.82 and y >= 0.82),
            "obstacle_x": bool(0.42 <= x <= 0.62),
            "obstacle_y": bool(0.42 <= y <= 0.62),
        }

    def atom_boundaries(self) -> dict[str, str]:
        return {"left": "x = 0.4", "bottom": "y = 0.4", "goal_zone": "x = 0.82 or y = 0.82", "obstacle_x": "x = 0.42/0.62", "obstacle_y": "y = 0.42/0.62"}

    def bounds_from_atom_assignment(self, atom_values: Mapping[str, bool], eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.bounds.low.copy(), self.bounds.high.copy()
        if atom_values.get("left") is True:
            high[0] = min(high[0], 0.4 - eps)
        elif atom_values.get("left") is False:
            low[0] = max(low[0], 0.4)
        if atom_values.get("bottom") is True:
            high[1] = min(high[1], 0.4 - eps)
        elif atom_values.get("bottom") is False:
            low[1] = max(low[1], 0.4)
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






class PendulumSwingUpEnvironment(ContinuousEnvironment):
    """连续力矩摆起任务，状态采用角度和角速度。"""

    def __init__(self) -> None:
        self.name = "pendulum_swing_up"
        self.state_names = ("theta", "theta_dot")
        self.action_names = ("力矩",)
        self.bounds = StateBounds(np.asarray([-np.pi, -8.0]), np.asarray([np.pi, 8.0]))
        self.dt = 0.05
        self.target_state = np.asarray([0.0, 0.0])
        self.actions = self.action_spec().initial_candidates()

    def action_spec(self) -> ContinuousActionSpec:
        return ContinuousActionSpec(
            low=np.asarray([-2.0]),
            high=np.asarray([2.0]),
            shape=(1,),
            critical_values=(-1.0, 1.0),
            max_candidates=9,
        )

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)])

    def nominal_initial_state(self) -> np.ndarray:
        return np.asarray([np.pi, 0.0])

    def safety_action(self) -> float:
        return 0.0

    def terminal_potential(self, state: ArrayLike) -> float:
        theta, theta_dot = self.bounds.clip(state)
        return float(theta**2 + 0.1 * theta_dot**2)

    def step(self, state: ArrayLike, action: float) -> StepResult:
        theta, theta_dot = self.bounds.clip(state)
        torque = self.nearest_action(action)
        next_theta_dot = float(np.clip(theta_dot + self.dt * (-3.0 * np.sin(theta) + torque), -8.0, 8.0))
        next_theta = float(((theta + self.dt * next_theta_dot + np.pi) % (2.0 * np.pi)) - np.pi)
        next_state = np.asarray([next_theta, next_theta_dot])
        cost = next_theta**2 + 0.1 * next_theta_dot**2 + 0.001 * torque**2
        done = bool(abs(next_theta) <= 0.08 and abs(next_theta_dot) <= 0.15)
        return StepResult(next_state, float(cost), done, {"terminal": "goal" if done else None, "transition_time": 1.0})

    def symbolic_atoms(self, variables: Mapping[str, Any]) -> dict[str, Any]:
        theta, theta_dot = variables["theta"], variables["theta_dot"]
        return {
            "left": theta < -0.5,
            "right": theta > 0.5,
            "moving_left": theta_dot < 0,
            "moving_right": theta_dot > 0,
            "upright": z3.And(theta >= -0.08, theta <= 0.08, theta_dot >= -0.15, theta_dot <= 0.15),
        }

    def symbolic_path_conditions(self, action: float) -> list[SymbolicPathCondition]:
        atoms = self.symbolic_atoms(self.make_symbolic_variables())
        action = self.nearest_action(action)
        rows = [
            ("upright", {"upright": True}, [atoms["upright"]], "已到达竖直稳定区。"),
            ("left", {"upright": False, "left": True}, [z3.Not(atoms["upright"]), atoms["left"]], "摆角位于左侧。"),
            ("right", {"upright": False, "right": True}, [z3.Not(atoms["upright"]), atoms["right"]], "摆角位于右侧。"),
            ("center_moving_left", {"upright": False, "left": False, "right": False, "moving_left": True}, [z3.Not(atoms["upright"]), z3.Not(atoms["left"]), z3.Not(atoms["right"]), atoms["moving_left"]], "中心区域且向左运动。"),
            ("center_moving_right", {"upright": False, "left": False, "right": False, "moving_left": False}, [z3.Not(atoms["upright"]), z3.Not(atoms["left"]), z3.Not(atoms["right"]), z3.Not(atoms["moving_left"])], "中心区域且未向左运动。"),
        ]
        return [SymbolicPathCondition(name, action, tuple(cons), values, desc) for name, values, cons, desc in rows]

    def evaluate_atoms(self, state: ArrayLike) -> dict[str, bool]:
        theta, theta_dot = self.bounds.clip(state)
        return {"left": bool(theta < -0.5), "right": bool(theta > 0.5), "moving_left": bool(theta_dot < 0), "moving_right": bool(theta_dot > 0), "upright": bool(abs(theta) <= 0.08 and abs(theta_dot) <= 0.15)}

    def atom_boundaries(self) -> dict[str, str]:
        return {"left": "theta = -0.5", "right": "theta = 0.5", "moving_left": "theta_dot = 0", "moving_right": "theta_dot = 0", "upright": "|theta| = 0.08, |theta_dot| = 0.15"}

    def bounds_from_atom_assignment(self, atom_values: Mapping[str, bool], eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.bounds.low.copy(), self.bounds.high.copy()
        if atom_values.get("upright") is True:
            low[0] = max(low[0], -0.08)
            high[0] = min(high[0], 0.08)
            low[1] = max(low[1], -0.15)
            high[1] = min(high[1], 0.15)
            return low, high
        if atom_values.get("left") is True:
            high[0] = min(high[0], -0.5 - eps)
        elif atom_values.get("left") is False:
            low[0] = max(low[0], -0.5)
        if atom_values.get("right") is True:
            low[0] = max(low[0], 0.5 + eps)
        elif atom_values.get("right") is False:
            high[0] = min(high[0], 0.5)
        if atom_values.get("moving_left") is True:
            high[1] = min(high[1], -eps)
        elif atom_values.get("moving_left") is False:
            low[1] = max(low[1], 0.0)
        if atom_values.get("moving_right") is True:
            low[1] = max(low[1], eps)
        elif atom_values.get("moving_right") is False:
            high[1] = min(high[1], 0.0)
        return low, high
