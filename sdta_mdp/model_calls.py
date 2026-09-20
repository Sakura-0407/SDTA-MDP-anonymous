from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator


class ModelCallCounter:
    """Count leaf dynamics evaluations only inside explicitly measured phases.

    The counter is local to an environment instance; do not share that instance
    across threads while measuring. Wrapper and base calls count once in total.
    """

    def __init__(self, env: Any) -> None:
        leaf = env
        seen: set[int] = set()
        while "base_env" in vars(leaf):
            if id(leaf) in seen:
                raise ValueError("Cyclic environment wrapper")
            seen.add(id(leaf))
            leaf = vars(leaf)["base_env"]
        self.env = leaf
        self.calls: Counter[str] = Counter()
        self.phase: str | None = None
        self._installed = False

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def __enter__(self) -> ModelCallCounter:
        if self._installed:
            raise RuntimeError("Counter is already installed")
        self._had_instance_step = "step" in vars(self.env)
        self._instance_step = vars(self.env).get("step")
        original = self.env.step

        @wraps(original)
        def counted_step(*args, **kwargs):
            if self.phase is not None:
                self.calls[self.phase] += 1
            return original(*args, **kwargs)

        self.env.step = counted_step
        self._installed = True
        return self

    def __exit__(self, *exc_info) -> None:
        if self._had_instance_step:
            self.env.step = self._instance_step
        else:
            del self.env.step
        self._installed = False
        self.phase = None

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self._installed:
            raise RuntimeError("Install counter before measuring")
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous
