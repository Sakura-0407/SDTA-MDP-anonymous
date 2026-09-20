from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import z3


@dataclass(frozen=True)
class SafeActionInterval:
    """具体连续状态上的安全动作闭区间。"""

    low: float
    high: float
    feasible: bool

    def clip(self, action: float) -> float:
        if not self.feasible:
            return float(action)
        return float(np.clip(action, self.low, self.high))


@dataclass(frozen=True)
class SymbolicActionSet:
    """动作变量与状态变量联合符号化得到的安全动作集合。"""

    name: str
    state_variables: Mapping[str, z3.ArithRef]
    action_variable: z3.ArithRef
    action_low: float
    action_high: float
    safe_formula: z3.BoolRef
    description: str

    def bounded_formula(self) -> z3.BoolRef:
        return z3.And(
            self.action_variable >= self.action_low,
            self.action_variable <= self.action_high,
            self.safe_formula,
        )

    def exists_projection(self) -> z3.BoolRef:
        """返回 `exists a. phi(x, a)`，用于状态级可行性证明。"""

        return z3.Exists([self.action_variable], self.bounded_formula())

    def eliminate_action(self) -> z3.BoolRef:
        """对线性实数算术尝试量词消除，得到仅含状态变量的公式。"""

        result = z3.Tactic("qe")(self.exists_projection())
        return z3.simplify(z3.And(*[goal.as_expr() for goal in result]))

    def interval_for_state(self, state_names: Sequence[str], state: Sequence[float] | np.ndarray) -> SafeActionInterval:
        """用 Optimize 求给定状态上的精确安全动作上下界。"""

        state_arr = np.asarray(state, dtype=float)
        substitutions = [
            (self.state_variables[name], z3.RealVal(str(float(state_arr[index]))))
            for index, name in enumerate(state_names)
        ]
        formula = z3.substitute(self.bounded_formula(), *substitutions)
        low = _optimize_bound(formula, self.action_variable, minimize=True)
        high = _optimize_bound(formula, self.action_variable, minimize=False)
        if low is None or high is None:
            return SafeActionInterval(self.action_low, self.action_high, False)
        return SafeActionInterval(low, high, True)

    def contains(self, state_names: Sequence[str], state: Sequence[float] | np.ndarray, action: float) -> bool:
        interval = self.interval_for_state(state_names, state)
        return interval.feasible and interval.low - 1e-9 <= action <= interval.high + 1e-9


def _optimize_bound(formula: z3.BoolRef, variable: z3.ArithRef, *, minimize: bool) -> float | None:
    optimizer = z3.Optimize()
    optimizer.add(formula)
    handle = optimizer.minimize(variable) if minimize else optimizer.maximize(variable)
    if optimizer.check() != z3.sat:
        return None
    return _z3_bound_to_float(optimizer.lower(handle) if minimize else optimizer.upper(handle))


def _z3_bound_to_float(value: Any) -> float:
    if z3.is_rational_value(value):
        return value.numerator_as_long() / value.denominator_as_long()
    text = str(value)
    if text.endswith("?"):
        text = text[:-1]
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)
