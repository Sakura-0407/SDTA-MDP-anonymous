from __future__ import annotations

from html import escape
import csv
from pathlib import Path
import numpy as np

from .environment import ContinuousEnvironment
from .solver import SolverResult


def write_solver_visualizations(
    out_dir: Path,
    env: ContinuousEnvironment,
    result: SolverResult,
    policy,
    *,
    seed: int,
) -> None:
    """为二维状态任务输出符号块、策略热图和 frontier 图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{env.name}_seed{seed}"
    _write_partition_svg(out_dir / f"{stem}_partition.svg", env, result)
    del policy
    _write_policy_svg(out_dir / f"{stem}_policy.svg", env, result)
    _write_frontier_dot(out_dir / f"{stem}_frontier.dot", result)
    if env.supports_exact_action_projection():
        _write_safe_action_interval_svg(out_dir / f"{stem}_safe_actions.svg", out_dir / f"{stem}_safe_actions.csv", env)


def _write_partition_svg(path: Path, env: ContinuousEnvironment, result: SolverResult) -> None:
    width, height, margin = 720, 560, 54
    low, high = env.bounds.low, env.bounds.high
    sx = (width - 2 * margin) / max(float(high[0] - low[0]), 1e-12)
    sy = (height - 2 * margin) / max(float(high[1] - low[1]), 1e-12)
    colors = {"core": "#7fcdbb", "edge": "#ef8a62", "remote": "#bdbdbd"}
    lines = _svg_header(width, height, f"{env.name}：符号分区")
    for block in result.blocks:
        x = margin + (float(block.lower[0]) - low[0]) * sx
        y = height - margin - (float(block.upper[1]) - low[1]) * sy
        w = max(2.0, (float(block.upper[0] - block.lower[0])) * sx)
        h = max(2.0, (float(block.upper[1] - block.lower[1])) * sy)
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{colors.get(block.role, "#ddd")}" fill-opacity="0.42" stroke="#333"/>')
        lines.append(f'<text x="{x + 3:.2f}" y="{y + 13:.2f}" font-size="10">B{block.block_id}</text>')
    lines.extend(_svg_axes(width, height, margin, env.state_names))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_policy_svg(path: Path, env: ContinuousEnvironment, result: SolverResult) -> None:
    """渲染块级策略先验，避免为绘图重复运行昂贵的局部 lookahead。"""

    width, height, margin, bins = 720, 560, 54, 24
    low, high = env.bounds.low, env.bounds.high
    xs = np.linspace(float(low[0]), float(high[0]), bins)
    ys = np.linspace(float(low[1]), float(high[1]), bins)
    cell_w, cell_h = (width - 2 * margin) / bins, (height - 2 * margin) / bins
    action_low, action_high = env.action_bounds()
    denominator = max(float(action_high[0] - action_low[0]), 1e-12)
    lines = _svg_header(width, height, f"{env.name}：策略热图")
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            state = np.asarray([x, y])
            block = min(result.blocks, key=lambda item: float(np.linalg.norm(item.witness - state)))
            action = result.action_for_block(block.block_id)
            ratio = float(np.clip((action - action_low[0]) / denominator, 0.0, 1.0))
            red, blue = int(230 * ratio), int(230 * (1.0 - ratio))
            px, py = margin + ix * cell_w, height - margin - (iy + 1) * cell_h
            lines.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cell_w + 0.2:.2f}" height="{cell_h + 0.2:.2f}" fill="rgb({red},100,{blue})"/>')
    lines.extend(_svg_axes(width, height, margin, env.state_names))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_frontier_dot(path: Path, result: SolverResult) -> None:
    lines = ["digraph Frontier {", '  rankdir="LR";']
    for block in result.blocks:
        lines.append(f'  B{block.block_id} [label="B{block.block_id}\\n{block.role}"];')
    for transition in result.partition_report.frontier_transitions:
        label = escape(f"a={transition.action:.3g}; {transition.boundary}")
        lines.append(f'  B{transition.source_block} -> B{transition.target_block} [label="{label}"];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_safe_action_interval_svg(svg_path: Path, csv_path: Path, env: ContinuousEnvironment) -> None:
    """沿首个状态维度绘制完全符号化动作区间。"""

    width, height, margin = 720, 420, 54
    low, high = env.bounds.low, env.bounds.high
    action_low, action_high = env.action_bounds()
    states = np.linspace(float(low[0]), float(high[0]), 80)
    base = env.nominal_initial_state().copy()
    rows: list[tuple[float, float, float, bool]] = []
    for value in states:
        state = base.copy()
        state[0] = value
        interval = env.exact_safe_action_interval(state)
        rows.append((value, interval.low if interval else 0.0, interval.high if interval else 0.0, bool(interval and interval.feasible)))

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([env.state_names[0], "safe_action_low", "safe_action_high", "feasible"])
        writer.writerows(rows)

    sx = (width - 2 * margin) / max(float(high[0] - low[0]), 1e-12)
    sy = (height - 2 * margin) / max(float(action_high[0] - action_low[0]), 1e-12)
    lines = _svg_header(width, height, f"{env.name}：完全符号化安全动作区间")
    upper_points, lower_points = [], []
    for state_value, interval_low, interval_high, feasible in rows:
        if not feasible:
            continue
        x = margin + (state_value - low[0]) * sx
        upper_points.append(f"{x:.2f},{height - margin - (interval_high - action_low[0]) * sy:.2f}")
        lower_points.append(f"{x:.2f},{height - margin - (interval_low - action_low[0]) * sy:.2f}")
    if upper_points:
        polygon = " ".join([*upper_points, *reversed(lower_points)])
        lines.append(f'<polygon points="{polygon}" fill="#7fcdbb" fill-opacity="0.55" stroke="#287271"/>')
    lines.extend(_svg_axes(width, height, margin, (env.state_names[0], "action")))
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="18" y="28" font-size="18">{escape(title)}</text>']


def _svg_axes(width: int, height: int, margin: int, names: tuple[str, ...]) -> list[str]:
    return [
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>',
        f'<text x="{width // 2}" y="{height - 12}" font-size="13">{escape(names[0])}</text>',
        f'<text x="8" y="{height // 2}" font-size="13">{escape(names[1])}</text>',
    ]
