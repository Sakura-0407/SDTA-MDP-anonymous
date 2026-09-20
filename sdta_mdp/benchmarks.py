from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .environment import BrakingCarEnvironment, ContinuousEnvironment, DoubleIntegratorParkingEnvironment, PointMassNavigationEnvironment
from .continuous_environments import ContinuousBrakingCarEnvironment, ContinuousDoubleIntegratorParkingEnvironment, ContinuousPointMassNavigationEnvironment, PendulumSwingUpEnvironment
from .stochastic_environments import StochasticActionEnvironment

@dataclass(frozen=True)
class BenchmarkSpec:
    """论文实验中一个 benchmark 的注册信息。"""
    name: str
    factory: Callable[[], ContinuousEnvironment]
    family: str
    description: str
    recommended_grid_bins: int
BENCHMARKS: dict[str, BenchmarkSpec] = {'braking_car': BenchmarkSpec('braking_car', BrakingCarEnvironment, 'safety_control', 'SymPar artifact 中的连续刹车车任务，目标是在墙前停住。', 10), 'point_mass': BenchmarkSpec('point_mass', PointMassNavigationEnvironment, 'navigation', '二维点质量导航任务，包含中心障碍区和目标区。', 10), 'double_integrator_parking': BenchmarkSpec('double_integrator_parking', DoubleIntegratorParkingEnvironment, 'control', '一维双积分器停车任务，要求位置和速度同时接近零。', 12), 'continuous_braking_car': BenchmarkSpec('continuous_braking_car', ContinuousBrakingCarEnvironment, 'continuous_safety_control', '连续制动力 braking car，目标是在墙前安全停住。', 10), 'pendulum_swing_up': BenchmarkSpec('pendulum_swing_up', PendulumSwingUpEnvironment, 'continuous_classic_control', '连续力矩摆起任务，要求摆角和角速度接近零。', 12), 'continuous_point_mass': BenchmarkSpec('continuous_point_mass', ContinuousPointMassNavigationEnvironment, 'continuous_navigation', '连续航向角二维导航任务，包含中心障碍区。', 10), 'continuous_double_integrator_parking': BenchmarkSpec('continuous_double_integrator_parking', ContinuousDoubleIntegratorParkingEnvironment, 'continuous_control', '连续加速度双积分器停车任务。', 12)}
STOCHASTIC_MAIN_BASES: dict[str, str] = {'stochastic_braking_car': 'braking_car', 'stochastic_point_mass': 'point_mass', 'stochastic_double_integrator_parking': 'double_integrator_parking', 'stochastic_continuous_braking_car': 'continuous_braking_car', 'stochastic_pendulum_swing_up': 'pendulum_swing_up', 'stochastic_continuous_point_mass': 'continuous_point_mass', 'stochastic_continuous_double_integrator_parking': 'continuous_double_integrator_parking'}
STOCHASTIC_MAIN_BENCHMARKS = tuple(STOCHASTIC_MAIN_BASES)
for _stochastic_name, _base_name in STOCHASTIC_MAIN_BASES.items():
    _base_spec = BENCHMARKS[_base_name]
    BENCHMARKS[_stochastic_name] = BenchmarkSpec(_stochastic_name, lambda base_factory=_base_spec.factory, stochastic_name=_stochastic_name: StochasticActionEnvironment(base_factory(), name=stochastic_name), f'stochastic_{_base_spec.family}', f'Action-stochastic version of {_base_name}.', _base_spec.recommended_grid_bins)

def make_benchmark(name: str) -> ContinuousEnvironment:
    if name not in BENCHMARKS:
        names = ', '.join(sorted(BENCHMARKS))
        raise ValueError(f'未知 benchmark：{name}。可选项：{names}')
    return BENCHMARKS[name].factory()

def benchmark_names() -> list[str]:
    return sorted(BENCHMARKS)
