from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import z3

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdta_mdp.benchmarks import make_benchmark
from sdta_mdp.symbolic import SymbolicBlock, SymbolicPartitioner


DEFAULT_BENCHMARKS = (
    "braking_car",
    "point_mass",
    "double_integrator_parking",
)
STOCHASTIC_BENCHMARKS = {
    benchmark: f"stochastic_{benchmark}" for benchmark in DEFAULT_BENCHMARKS
}
DEFAULT_UPSTREAM_ROOT = REPO_ROOT / "external" / "SymPar"


@dataclass(frozen=True)
class EnvironmentScala:
    actions: tuple[float, ...]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    reset_generator: str
    transition_body: str
    terminal_body: str
    success_body: str
    distance_body: str


def _number(value: float) -> str:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Cannot serialize non-finite value {value!r} to Scala.")
    rendered = repr(value)
    if "e" in rendered:
        base, exponent = rendered.split("e", 1)
        rendered = f"{base}e{int(exponent)}"
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered


def _z3_number(expr: z3.ArithRef) -> str:
    if z3.is_rational_value(expr):
        numerator = expr.numerator_as_long()
        denominator = expr.denominator_as_long()
        if denominator == 1:
            return _number(float(numerator))
        return f"({_number(float(numerator))} / {_number(float(denominator))})"
    if z3.is_algebraic_value(expr):
        return _number(float(expr.approx(20).as_decimal(20).rstrip("?")))
    raise TypeError(f"Unsupported Z3 numeric expression: {expr!r}")


def _z3_to_scala(expr: z3.ExprRef, variables: dict[str, str]) -> str:
    expr = z3.simplify(expr)
    if z3.is_true(expr):
        return "true"
    if z3.is_false(expr):
        return "false"
    if z3.is_rational_value(expr) or z3.is_algebraic_value(expr):
        return _z3_number(expr)
    if z3.is_const(expr) and expr.num_args() == 0:
        name = str(expr)
        if name not in variables:
            raise KeyError(f"Unknown Z3 variable {name!r}.")
        return variables[name]

    kind = expr.decl().kind()
    args = [_z3_to_scala(arg, variables) for arg in expr.children()]
    binary = {
        z3.Z3_OP_EQ: "==",
        z3.Z3_OP_LE: "<=",
        z3.Z3_OP_LT: "<",
        z3.Z3_OP_GE: ">=",
        z3.Z3_OP_GT: ">",
        z3.Z3_OP_SUB: "-",
        z3.Z3_OP_DIV: "/",
    }
    if kind in binary and len(args) == 2:
        return f"({args[0]} {binary[kind]} {args[1]})"
    if kind == z3.Z3_OP_AND:
        return "(" + " && ".join(args) + ")"
    if kind == z3.Z3_OP_OR:
        return "(" + " || ".join(args) + ")"
    if kind == z3.Z3_OP_NOT and len(args) == 1:
        return f"(!{args[0]})"
    if kind == z3.Z3_OP_ADD:
        return "(" + " + ".join(args) + ")"
    if kind == z3.Z3_OP_MUL:
        return "(" + " * ".join(args) + ")"
    if kind == z3.Z3_OP_UMINUS and len(args) == 1:
        return f"(-{args[0]})"
    if kind == z3.Z3_OP_TO_REAL and len(args) == 1:
        return args[0]
    raise TypeError(f"Unsupported Z3 expression for Scala translation: {expr!r}")


def _environment_scala(name: str) -> EnvironmentScala:
    if name == "braking_car":
        return EnvironmentScala(
            actions=(-10.0, -5.0, -2.5, -0.5, -0.05, -0.01, -0.001),
            bounds=((0.0, 15.0), (0.0, 20.0)),
            reset_generator="""
      x0 <- Randomized2.between(0.5, 10.0)
      x1 <- Randomized2.between(0.0, 8.5)
      state = SDTAState(x0, x1, 0)
    yield state""",
            transition_body="""
    val velocity = clip(s.x0, 0.0, 15.0)
    val position = clip(s.x1, 0.0, 20.0)
    val aa = nearestAction(a)
    if velocity <= 1.0e-9 then
      SDTATransition(SDTAState(0.0, position, s.t + 1), 0.0, true, "stopped", false)
    else if position >= 10.0 then
      SDTATransition(SDTAState(velocity, 10.0, s.t + 1), 100.0, true, "collision", true)
    else
      val stopTime = -velocity / aa
      val transitionTime = math.min(stopTime, 2.0)
      val rawPosition = position + velocity * transitionTime +
        0.5 * aa * transitionTime * transitionTime
      val nextPosition = math.min(rawPosition, 10.0)
      val nextVelocity =
        if nextPosition >= 10.0 then 0.0
        else math.max(velocity + aa * transitionTime, 0.0)
      val collision = nextPosition >= 10.0
      val stopped = nextVelocity <= 1.0e-9
      val cost = if collision then 100.0 else -aa + (if stopped then 0.0 else 0.05)
      val terminal =
        if collision then "collision"
        else if stopped then "stopped"
        else ""
      SDTATransition(
        SDTAState(nextVelocity, nextPosition, s.t + 1),
        cost,
        collision || stopped,
        terminal,
        collision,
      )""",
            terminal_body="s.x0 <= 1.0e-9 || s.x1 >= 10.0",
            success_body='terminal == "stopped" || distanceToTarget(s) < 0.1',
            distance_body="math.hypot(s.x0 - 0.0, s.x1 - 9.5)",
        )
    if name == "point_mass":
        return EnvironmentScala(
            actions=(0.0, 1.0, 2.0, 3.0, 4.0),
            bounds=((0.0, 1.0), (0.0, 1.0)),
            reset_generator="""
      x0 <- Randomized2.between(0.05, 0.2)
      x1 <- Randomized2.between(0.05, 0.2)
      state = SDTAState(x0, x1, 0)
    yield state""",
            transition_body="""
    val x = clip(s.x0, 0.0, 1.0)
    val y = clip(s.x1, 0.0, 1.0)
    val aa = nearestAction(a)
    val (dx, dy) =
      if aa == 0.0 then (-0.08, 0.0)
      else if aa == 1.0 then (0.08, 0.0)
      else if aa == 2.0 then (0.0, -0.08)
      else if aa == 3.0 then (0.0, 0.08)
      else (0.0, 0.0)
    val nextX = clip(x + dx, 0.0, 1.0)
    val nextY = clip(y + dy, 0.0, 1.0)
    val inObstacle =
      nextX >= 0.42 && nextX <= 0.62 && nextY >= 0.42 && nextY <= 0.62
    val distance = math.hypot(nextX - 0.9, nextY - 0.9)
    val done = distance <= 0.08
    val cost = distance + 0.02 + (if inObstacle then 20.0 else 0.0)
    SDTATransition(
      SDTAState(nextX, nextY, s.t + 1),
      cost,
      done,
      if done then "goal" else "",
      inObstacle,
    )""",
            terminal_body="math.hypot(s.x0 - 0.9, s.x1 - 0.9) <= 0.08",
            success_body='terminal == "goal" || distanceToTarget(s) < 0.1',
            distance_body="math.hypot(s.x0 - 0.9, s.x1 - 0.9)",
        )
    if name == "double_integrator_parking":
        return EnvironmentScala(
            actions=(-1.0, 0.0, 1.0),
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            reset_generator="""
      x0 <- Randomized2.between(-1.5, 1.5)
      x1 <- Randomized2.between(-0.6, 0.6)
      state = SDTAState(x0, x1, 0)
    yield state""",
            transition_body="""
    val position = clip(s.x0, -2.0, 2.0)
    val velocity = clip(s.x1, -2.0, 2.0)
    val aa = nearestAction(a)
    val nextVelocity = clip(velocity + 0.2 * aa, -2.0, 2.0)
    val nextPosition = clip(position + 0.2 * velocity + 0.02 * aa, -2.0, 2.0)
    val done = math.abs(nextPosition) <= 0.08 && math.abs(nextVelocity) <= 0.08
    val cost =
      nextPosition * nextPosition +
      0.2 * nextVelocity * nextVelocity +
      0.05 * aa * aa
    SDTATransition(
      SDTAState(nextPosition, nextVelocity, s.t + 1),
      cost,
      done,
      if done then "parked" else "",
      false,
    )""",
            terminal_body="math.abs(s.x0) <= 0.08 && math.abs(s.x1) <= 0.08",
            success_body='terminal == "parked" || distanceToTarget(s) < 0.1',
            distance_body="math.hypot(s.x0, s.x1)",
        )
    raise ValueError(f"No upstream SymPar adapter is defined for {name!r}.")


def _partition_scala(
    env: Any,
    blocks: Iterable[SymbolicBlock],
) -> tuple[str, str]:
    variable_names = {name: f"s.x{idx}" for idx, name in enumerate(env.state_names)}
    partition_rows: list[str] = []
    witness_rows: list[str] = []
    for block in blocks:
        predicate = _z3_to_scala(block.formula(), variable_names)
        partition_rows.append(f"  ({block.block_id}, s => {predicate})")
        witness_rows.append(
            "  SDTAState("
            f"{_number(block.witness[0])}, {_number(block.witness[1])}, 0)"
        )
    return (
        "List(\n" + ",\n".join(partition_rows) + "\n)",
        "Vector(\n" + ",\n".join(witness_rows) + "\n)",
    )


def _control_refinement_predicates(
    benchmark: str,
    partitioner: SymbolicPartitioner,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    x0 = partitioner.variables[partitioner.env.state_names[0]]
    x1 = partitioner.variables[partitioner.env.state_names[1]]
    if benchmark == "braking_car":
        return ()
    if benchmark == "point_mass":
        return (
            ("x_clears_obstacle", x0 >= z3.RealVal("0.62")),
            ("y_clears_obstacle", x1 >= z3.RealVal("0.62")),
            ("x_reaches_target_band", x0 >= z3.RealVal("0.82")),
            ("y_reaches_target_band", x1 >= z3.RealVal("0.82")),
        )
    if benchmark == "double_integrator_parking":
        return (
            ("position_nonnegative", x0 >= 0),
            ("velocity_nonnegative", x1 >= 0),
        )
    raise ValueError(f"No control refinement is defined for {benchmark!r}.")


def _refine_blocks_for_control(
    *,
    benchmark: str,
    partitioner: SymbolicPartitioner,
    blocks: tuple[SymbolicBlock, ...],
) -> tuple[SymbolicBlock, ...]:
    predicates = _control_refinement_predicates(benchmark, partitioner)
    if not predicates:
        return blocks

    refined: list[SymbolicBlock] = []
    for block in blocks:
        candidates: list[
            tuple[tuple[z3.BoolRef, ...], dict[str, bool], tuple[str, ...]]
        ] = [(block.constraints, dict(block.atom_values), block.descriptions)]
        for name, predicate in predicates:
            next_candidates = []
            for constraints, atom_values, descriptions in candidates:
                for value, literal in (
                    (True, predicate),
                    (False, z3.Not(predicate)),
                ):
                    refined_constraints = constraints + (literal,)
                    witness = partitioner._witness(refined_constraints)
                    if witness is None:
                        continue
                    next_candidates.append(
                        (
                            refined_constraints,
                            {**atom_values, f"control_refinement:{name}": value},
                            descriptions + (f"{name}={value}",),
                        )
                    )
            candidates = next_candidates

        for constraints, atom_values, descriptions in candidates:
            witness = partitioner._witness(constraints)
            if witness is None:
                continue
            refined.append(
                SymbolicBlock(
                    block_id=len(refined),
                    constraints=constraints,
                    action_paths=dict(block.action_paths),
                    atom_values=atom_values,
                    descriptions=descriptions,
                    witness=np.asarray(witness, dtype=float),
                    role=block.role,
                    frontier_boundaries=block.frontier_boundaries,
                    lower=block.lower.copy(),
                    upper=block.upper.copy(),
                )
            )
    return tuple(refined)


def _agent_source(
    *,
    benchmark: str,
    env_spec: EnvironmentScala,
    partitions: str,
    witnesses: str,
    block_count: int,
    max_steps: int,
    gamma: float,
    reward_mode: str,
    success_bonus: float,
    potential_scale: float,
    training_initialization: str,
    slip_probability: float,
) -> str:
    action_values = ", ".join(_number(value) for value in env_spec.actions)
    low0, high0 = env_spec.bounds[0]
    low1, high1 = env_spec.bounds[1]
    if reward_mode == "matched-cost":
        reward_expression = "-result.cost"
    elif reward_mode == "terminal-bonus":
        reward_expression = (
            "-result.cost + "
            f"(if result.done && isSuccess(result.next, result.terminal) "
            f"then {_number(success_bonus)} else 0.0)"
        )
    elif reward_mode == "potential-shaped":
        reward_expression = (
            "-result.cost + "
            f"{_number(potential_scale)} * "
            f"({_number(gamma)} * "
            "(if result.done then 0.0 else -distanceToTarget(result.next)) "
            "- (-distanceToTarget(s)))"
        )
    else:
        raise ValueError(f"Unsupported reward mode {reward_mode!r}.")
    if training_initialization == "mixed-witness":
        initialize_expression = """for
      selected <- Randomized2.between(0, P_NUM + 2)
      state <-
        if selected < P_NUM then Randomized2.const(witnesses(selected))
        else resetGenerator
    yield state"""
    elif training_initialization == "reset-only":
        initialize_expression = "resetGenerator"
    else:
        raise ValueError(
            f"Unsupported training initialization {training_initialization!r}."
        )
    if slip_probability > 0.0:
        step_body = f"""    val intended = nearestAction(a)
    val alternatives = actions.filter(candidate => candidate != intended)
    for
      precise <- Randomized2.coin({_number(1.0 - slip_probability)})
      executed <-
        if precise || alternatives.isEmpty then Randomized2.const(intended)
        else Randomized2.oneOf(alternatives*)
      result = transition(s, executed)
      reward = {reward_expression}
    yield (result.next, reward)"""
    else:
        step_body = f"""    val result = transition(s, a)
    val reward = {reward_expression}
    Randomized2.const((result.next, reward))"""
    return f"""import symsim.*
import symsim.concrete.Randomized2
import cats.syntax.all.*

case class SDTAState(x0: Double, x1: Double, t: Int)
case class SDTATransition(
  next: SDTAState,
  cost: Double,
  done: Boolean,
  terminal: String,
  violation: Boolean,
)

type SDTAObservableState = Int
type SDTAAction = Double
type SDTAReward = Double

val benchmarkName: String = "{benchmark}"
val P_NUM: Int = {block_count}
val actions: Vector[Double] = Vector({action_values})
val partitions: List[(Int, SDTAState => Boolean)] = {partitions}
val witnesses: Vector[SDTAState] = {witnesses}

class SDTAAgent(using probula.RNG)
  extends Agent[
    SDTAState,
    SDTAObservableState,
    SDTAAction,
    SDTAReward,
    Randomized2,
  ]
  with Episodic:

  val TimeHorizon: Int = {max_steps}
  val instances = new SDTAInstances
  var modelCalls: Long = 0L

  def clip(value: Double, low: Double, high: Double): Double =
    math.max(low, math.min(high, value))

  def nearestAction(value: Double): Double =
    actions.minBy(candidate => math.abs(candidate - value))

  def observe(s: SDTAState): SDTAObservableState =
    partitions
      .find(_._2(s))
      .map(_._1)
      .getOrElse(
        throw IllegalStateException(
          s"No supplied symbolic block contains state $s in $benchmarkName"
        )
      )

  def environmentTerminal(s: SDTAState): Boolean =
    {env_spec.terminal_body}

  def isFinal(s: SDTAState): Boolean =
    environmentTerminal(s) || s.t >= TimeHorizon

  def distanceToTarget(s: SDTAState): Double =
    {env_spec.distance_body}

  def isSuccess(s: SDTAState, terminal: String): Boolean =
    {env_spec.success_body}

  def transition(s: SDTAState, a: SDTAAction): SDTATransition =
    modelCalls += 1L
{env_spec.transition_body}

  def step(s: SDTAState)(a: SDTAAction): Randomized2[(SDTAState, SDTAReward)] =
{step_body}

  def resetGenerator: Randomized2[SDTAState] =
    for
{env_spec.reset_generator}

  def initialize: Randomized2[SDTAState] =
    {initialize_expression}

end SDTAAgent

class SDTAInstances(using probula.RNG)
  extends AgentConstraints[
    SDTAState,
    SDTAObservableState,
    SDTAAction,
    SDTAReward,
    Randomized2,
  ]:

  import cats.{{Eq, Monad}}
  import cats.kernel.BoundedEnumerable
  import org.scalacheck.{{Arbitrary, Gen}}

  given enumAction: BoundedEnumerable[SDTAAction] =
    BoundedEnumerableFromList(actions*)

  given enumState: BoundedEnumerable[SDTAObservableState] =
    BoundedEnumerableFromList((0 until P_NUM)*)

  given schedulerIsMonad: Monad[Randomized2] =
    symsim.concrete.Randomized2.randomizedIsMonad

  given canTestInScheduler: CanTestIn[Randomized2] =
    symsim.concrete.Randomized2.canTestInRandomized

  lazy val genState: Gen[SDTAState] = for
    x0 <- Gen.choose({_number(low0)}, {_number(high0)})
    x1 <- Gen.choose({_number(low1)}, {_number(high1)})
  yield SDTAState(x0, x1, 0)

  given arbitraryState: Arbitrary[SDTAState] = Arbitrary(genState)
  given eqState: Eq[SDTAState] = Eq.fromUniversalEquals
  given arbitraryReward: Arbitrary[SDTAReward] = Arbitrary(Gen.double)
  given rewardArith: Arith[SDTAReward] = Arith.arithDouble

end SDTAInstances
"""


def _initial_states(env: Any, seeds: Iterable[int], episodes: int) -> dict[int, list[np.ndarray]]:
    output: dict[int, list[np.ndarray]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        output[int(seed)] = [np.asarray(env.reset(rng), dtype=float) for _ in range(episodes)]
    return output


def _experiment_source(
    *,
    benchmark: str,
    initial_states: dict[int, list[np.ndarray]],
    training_episodes: int,
    max_steps: int,
    alpha: float,
    gamma: float,
    epsilon0: float,
    epsilon_decay_factor: float,
    min_exploration: float,
) -> str:
    seed_rows: list[str] = []
    for seed, rows in initial_states.items():
        states = ", ".join(
            f"SDTAState({_number(row[0])}, {_number(row[1])}, 0)" for row in rows
        )
        seed_rows.append(f"  {seed} -> List({states})")
    seeds_scala = "Map(\n" + ",\n".join(seed_rows) + "\n)"
    return f"""import java.io.PrintWriter
import symsim.concrete.ConcreteQLearningWithDecay
import spire.random.rng.MersenneTwister64

case class EpisodeResult(
  episode: Int,
  cost: Double,
  success: Boolean,
  violation: Boolean,
  steps: Int,
  terminal: String,
  finalX0: Double,
  finalX1: Double,
)

val evaluationInitials: Map[Int, List[SDTAState]] = {seeds_scala}

def evaluatePolicy(
  agent: SDTAAgent,
  policy: scala.collection.immutable.Map[SDTAObservableState, SDTAAction],
  initials: List[SDTAState],
): List[EpisodeResult] =
  initials.zipWithIndex.map {{ case (initial, episode) =>
    var state = initial
    var totalCost = 0.0
    var violation = false
    var terminal = ""
    var steps = 0
    while steps < {max_steps} && !agent.isFinal(state) do
      val action = policy(agent.observe(state))
      val transition = agent.transition(state, action)
      totalCost += transition.cost
      violation = violation || transition.violation
      terminal = transition.terminal
      state = transition.next
      steps += 1
    EpisodeResult(
      episode,
      totalCost,
      agent.isSuccess(state, terminal),
      violation || terminal == "collision",
      steps,
      terminal,
      state.x0,
      state.x1,
    )
  }}

def writeEpisodeResults(seed: Int, rows: List[EpisodeResult]): Unit =
  val writer = new PrintWriter(s"episodes_seed_$seed.csv")
  try
    writer.println(
      "episode,cost,success,violation,steps,terminal,final_x0,final_x1"
    )
    rows.foreach {{ row =>
      writer.println(
        s"${{row.episode}},${{row.cost}},${{row.success}},${{row.violation}}," +
        s"${{row.steps}},${{row.terminal}},${{row.finalX0}},${{row.finalX1}}"
      )
    }}
  finally writer.close()

def runSeed(seed: Int): Unit =
  given MersenneTwister64 = MersenneTwister64.fromTime(seed.toLong + 1L)
  val agent = new SDTAAgent
  import agent.instances.{{enumAction, enumState}}

  val setup = new ConcreteQLearningWithDecay(
    agent = agent,
    alpha = {_number(alpha)},
    gamma = {_number(gamma)},
    epsilon0 = {_number(epsilon0)},
    episodes = {training_episodes},
  ):
    override def decayFactor: Double = {_number(epsilon_decay_factor)}
    override def minExploration: Double = {_number(min_exploration)}

  val trainingStart = System.nanoTime()
  val learnedPolicy = setup.run
  val finalPolicy = (0 until P_NUM)
    .map(block => block -> learnedPolicy.getOrElse(block, actions.head))
    .toMap
  val defaultedBlockIds = (0 until P_NUM).filterNot(learnedPolicy.contains)
  val defaultedBlocks = defaultedBlockIds.length
  val trainingSeconds = (System.nanoTime() - trainingStart).toDouble / 1.0e9
  val trainingCalls = agent.modelCalls

  val evaluationStart = System.nanoTime()
  val episodeRows = evaluatePolicy(agent, finalPolicy, evaluationInitials(seed))
  val evaluationSeconds = (System.nanoTime() - evaluationStart).toDouble / 1.0e9
  val evaluationCalls = agent.modelCalls - trainingCalls

  writeEpisodeResults(seed, episodeRows)
  val meanCost = episodeRows.map(_.cost).sum / episodeRows.length.toDouble
  val successRate = episodeRows.count(_.success).toDouble / episodeRows.length.toDouble
  val violationRate =
    episodeRows.count(_.violation).toDouble / episodeRows.length.toDouble
  val policyJson = (0 until P_NUM)
    .map(block => "\\\"" + block + "\\\":" + finalPolicy(block))
    .mkString("{{", ",", "}}")
  val defaultedBlockIdsJson = defaultedBlockIds.mkString("[", ",", "]")
  val json =
    s\"\"\"{{"benchmark":"{benchmark}","method":"SymPar-Q-Upstream","seed":$seed,"training_episodes":{training_episodes},"evaluation_episodes":${{episodeRows.length}},"training_seconds":$trainingSeconds,"evaluation_seconds":$evaluationSeconds,"training_model_calls":$trainingCalls,"evaluation_model_calls":$evaluationCalls,"mean_cost":$meanCost,"success_rate":$successRate,"violation_rate":$violationRate,"blocks":$P_NUM,"learned_policy_entries":${{learnedPolicy.size}},"defaulted_blocks":$defaultedBlocks,"defaulted_block_ids":$defaultedBlockIdsJson,"policy":$policyJson}}\"\"\"
  val writer = new PrintWriter(s"result_seed_$seed.json")
  try writer.println(json)
  finally writer.close()
  println(
    s"RESULT benchmark={benchmark} seed=$seed cost=$meanCost " +
    s"success=$successRate violation=$violationRate " +
    s"train_s=$trainingSeconds eval_s=$evaluationSeconds"
  )

@main def run(seeds: Int*): Unit =
  val requested = if seeds.nonEmpty then seeds else evaluationInitials.keys.toSeq.sorted
  requested.foreach(runSeed)
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_workdir(
    *,
    benchmark: str,
    blocks: tuple[SymbolicBlock, ...],
    env: Any,
    env_spec: EnvironmentScala,
    upstream_root: Path,
    workdir: Path,
    seeds: tuple[int, ...],
    training_episodes: int,
    evaluation_episodes: int,
    max_steps: int,
    alpha: float,
    gamma: float,
    epsilon0: float,
    epsilon_decay_factor: float,
    min_exploration: float,
    reward_mode: str,
    success_bonus: float,
    potential_scale: float,
    partition_variant: str,
    training_initialization: str,
    slip_probability: float,
) -> None:
    if workdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing work directory: {workdir}")
    workdir.mkdir(parents=True)
    for filename in (
        "symsim.jar",
        "project.scala",
    ):
        shutil.copy2(upstream_root / "symsim-files" / filename, workdir / filename)

    partitions, witnesses = _partition_scala(env, blocks)
    (workdir / "SDTAAgent.scala").write_text(
        _agent_source(
            benchmark=benchmark,
            env_spec=env_spec,
            partitions=partitions,
            witnesses=witnesses,
            block_count=len(blocks),
            max_steps=max_steps,
            gamma=gamma,
            reward_mode=reward_mode,
            success_bonus=success_bonus,
            potential_scale=potential_scale,
            training_initialization=training_initialization,
            slip_probability=slip_probability,
        ),
        encoding="utf-8",
    )
    initials = _initial_states(env, seeds, evaluation_episodes)
    (workdir / "Experiments.scala").write_text(
        _experiment_source(
            benchmark=benchmark,
            initial_states=initials,
            training_episodes=training_episodes,
            max_steps=max_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon0=epsilon0,
            epsilon_decay_factor=epsilon_decay_factor,
            min_exploration=min_exploration,
        ),
        encoding="utf-8",
    )
    manifest = {
        "benchmark": benchmark,
        "seeds": list(seeds),
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon0": epsilon0,
        "epsilon_decay_factor": epsilon_decay_factor,
        "min_exploration": min_exploration,
        "reward_mode": reward_mode,
        "success_bonus": success_bonus,
        "potential_scale": potential_scale,
        "partition_variant": partition_variant,
        "training_initialization": training_initialization,
        "transition_regime": "stochastic" if slip_probability > 0.0 else "deterministic",
        "slip_probability": slip_probability,
        "stochastic_training_backend": (
            "official symsim Randomized2.coin/oneOf"
            if slip_probability > 0.0
            else "not applicable"
        ),
        "upstream_record": "https://doi.org/10.5281/zenodo.14620119",
        "upstream_symsim_sha256": _sha256(upstream_root / "symsim-files" / "symsim.jar"),
        "upstream_project_sha256": _sha256(upstream_root / "symsim-files" / "project.scala"),
        "adapter_source_sha256": _sha256(workdir / "SDTAAgent.scala"),
        "experiment_source_sha256": _sha256(workdir / "Experiments.scala"),
        "partition_blocks": len(blocks),
        "partition_witnesses": [block.witness.tolist() for block in blocks],
    }
    (workdir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _python_episode_rows(
    *,
    env: Any,
    partitioner: SymbolicPartitioner,
    blocks: tuple[SymbolicBlock, ...],
    policy: dict[str, float],
    seed: int,
    episodes: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    base_benchmark = (
        env.base_benchmark_name()
        if callable(getattr(env, "base_benchmark_name", None))
        else env.name
    )

    def is_environment_terminal(state: np.ndarray) -> bool:
        if base_benchmark == "braking_car":
            return bool(state[0] <= 1e-9 or state[1] >= 10.0)
        if base_benchmark == "point_mass":
            return bool(env.distance_to_target(state) <= 0.08)
        if base_benchmark == "double_integrator_parking":
            return bool(abs(state[0]) <= 0.08 and abs(state[1]) <= 0.08)
        raise ValueError(f"Unsupported parity environment: {env.name}")

    rng = np.random.default_rng(seed)
    reset_action_diagnostics = getattr(env, "reset_evaluation_action_diagnostics", None)
    record_evaluation_action = getattr(env, "record_evaluation_action", None)
    if callable(reset_action_diagnostics):
        reset_action_diagnostics()
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        state = np.asarray(env.reset(rng), dtype=float)
        total_cost = 0.0
        violation = False
        terminal = ""
        steps = 0
        for _ in range(max_steps):
            if is_environment_terminal(state):
                break
            block = partitioner.block_for_state(state, blocks)
            if block is None:
                raise RuntimeError(
                    f"No supplied block contains {env.name} state {state.tolist()}."
                )
            action = float(policy[str(block.block_id)])
            result = env.step(state, action)
            if callable(record_evaluation_action):
                record_evaluation_action(result.info)
            total_cost += float(result.cost)
            violation = violation or bool(
                result.info.get("constraint_violation", False)
            )
            terminal = str(result.info.get("terminal") or "")
            state = np.asarray(result.next_state, dtype=float)
            steps += 1
            if result.done:
                break
        success = bool(
            terminal in {"goal", "stopped", "parked"}
            or env.distance_to_target(state) < 0.1
        )
        rows.append(
            {
                "episode": episode,
                "cost": total_cost,
                "success": success,
                "violation": bool(violation or terminal == "collision"),
                "steps": steps,
                "terminal": terminal,
                "final_x0": float(state[0]),
                "final_x1": float(state[1]),
            }
        )
    return rows


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "episode",
        "cost",
        "success",
        "violation",
        "steps",
        "terminal",
        "final_x0",
        "final_x1",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_episode_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "episode": int(row["episode"]),
            "cost": float(row["cost"]),
            "success": row["success"].lower() == "true",
            "violation": row["violation"].lower() == "true",
            "steps": int(row["steps"]),
            "terminal": row["terminal"],
            "final_x0": float(row["final_x0"]),
            "final_x1": float(row["final_x1"]),
        }
        for row in rows
    ]


def _validate_episode_parity(
    scala_rows: list[dict[str, Any]],
    python_rows: list[dict[str, Any]],
    *,
    tolerance: float = 1e-8,
) -> None:
    if len(scala_rows) != len(python_rows):
        raise AssertionError(
            f"Scala/Python evaluation row count differs: "
            f"{len(scala_rows)} != {len(python_rows)}"
        )
    for scala_row, python_row in zip(scala_rows, python_rows):
        for key in ("episode", "success", "violation", "steps", "terminal"):
            if scala_row[key] != python_row[key]:
                raise AssertionError(
                    f"Scala/Python mismatch for episode {scala_row['episode']} "
                    f"field {key}: {scala_row[key]!r} != {python_row[key]!r}"
                )
        for key in ("cost", "final_x0", "final_x1"):
            if not np.isclose(
                float(scala_row[key]),
                float(python_row[key]),
                rtol=tolerance,
                atol=tolerance,
            ):
                raise AssertionError(
                    f"Scala/Python mismatch for episode {scala_row['episode']} "
                    f"field {key}: {scala_row[key]!r} != {python_row[key]!r}"
                )


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "method",
        "seed",
        "seconds",
        "preparation_seconds",
        "evaluation_seconds",
        "training_seconds",
        "partition_seconds",
        "scala_process_seconds",
        "training_episodes",
        "evaluation_episodes",
        "iterations",
        "alpha",
        "gamma",
        "epsilon0",
        "epsilon_decay_factor",
        "min_exploration",
        "reward_mode",
        "success_bonus",
        "potential_scale",
        "partition_variant",
        "training_initialization",
        "mean_cost",
        "success_rate",
        "violation_rate",
        "blocks",
        "model_calls",
        "model_call_accounting",
        "training_model_calls",
        "evaluation_model_calls",
        "learned_policy_entries",
        "defaulted_blocks",
        "defaulted_block_ids",
        "status",
        "source_artifact",
        "partition_source",
        "environment_parity",
        "transition_regime",
        "noise_model",
        "noise_scale",
        "action_sample_count",
        "action_perturbation_count",
        "action_perturbation_rate",
        "mean_abs_action_deviation",
        "mean_intended_action",
        "mean_executed_action",
        "evaluation_owner",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    upstream_root = args.upstream_root.resolve()
    if not (upstream_root / "symsim-files" / "symsim.jar").is_file():
        raise FileNotFoundError(
            f"Official SymPar symsim.jar was not found under {upstream_root}."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for base_benchmark in args.benchmarks:
        benchmark = (
            STOCHASTIC_BENCHMARKS[base_benchmark]
            if args.stochastic_slip_probability > 0.0
            else base_benchmark
        )
        partition_env = make_benchmark(base_benchmark)
        evaluation_env = make_benchmark(benchmark)
        partitioner = SymbolicPartitioner(partition_env, frontier_mode="sampled", seed=0)
        partition_start = perf_counter()
        blocks, _ = partitioner.build_partition()
        if args.partition_variant == "control-refined":
            blocks = _refine_blocks_for_control(
                benchmark=base_benchmark,
                partitioner=partitioner,
                blocks=blocks,
            )
        partition_seconds = perf_counter() - partition_start
        env_spec = _environment_scala(base_benchmark)
        workdir = args.out_dir / benchmark
        _write_workdir(
            benchmark=benchmark,
            blocks=blocks,
            env=evaluation_env,
            env_spec=env_spec,
            upstream_root=upstream_root,
            workdir=workdir,
            seeds=args.seeds,
            training_episodes=args.training_episodes,
            evaluation_episodes=args.evaluation_episodes,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon0=args.epsilon0,
            epsilon_decay_factor=args.epsilon_decay_factor,
            min_exploration=args.min_exploration,
            reward_mode=args.reward_mode,
            success_bonus=args.success_bonus,
            potential_scale=args.potential_scale,
            partition_variant=args.partition_variant,
            training_initialization=args.training_initialization,
            slip_probability=args.stochastic_slip_probability,
        )
        command = [
            args.scala_cli,
            "run",
            ".",
            "--server=false",
            "--scala",
            args.scala_version,
            "--",
            *(str(seed) for seed in args.seeds),
        ]
        process_start = perf_counter()
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        scala_process_seconds = perf_counter() - process_start
        (workdir / "scala_stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (workdir / "scala_stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        (workdir / "scala_exit_code.txt").write_text(
            f"{completed.returncode}\n",
            encoding="ascii",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Upstream SymPar Scala run failed for {benchmark} with exit "
                f"code {completed.returncode}. See {workdir / 'scala_stderr.log'}."
            )

        for seed in args.seeds:
            result_path = workdir / f"result_seed_{seed}.json"
            episode_path = workdir / f"episodes_seed_{seed}.csv"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            scala_episode_rows = _read_episode_csv(episode_path)
            python_env = make_benchmark(benchmark)
            python_evaluation_start = perf_counter()
            python_episode_rows = _python_episode_rows(
                env=python_env,
                partitioner=partitioner,
                blocks=blocks,
                policy={str(key): float(value) for key, value in result["policy"].items()},
                seed=seed,
                episodes=args.evaluation_episodes,
                max_steps=args.max_steps,
            )
            python_evaluation_seconds = perf_counter() - python_evaluation_start
            if args.stochastic_slip_probability > 0.0:
                shutil.move(
                    episode_path,
                    workdir / f"scala_episodes_seed_{seed}.csv",
                )
                _write_episode_csv(episode_path, python_episode_rows)
                environment_parity = (
                    "official Randomized2 stochastic training; "
                    "Python unified stochastic-protocol evaluation"
                )
                evaluation_seconds = python_evaluation_seconds
                evaluation_calls = sum(int(row["steps"]) for row in python_episode_rows)
                mean_cost = float(np.mean([float(row["cost"]) for row in python_episode_rows]))
                success_rate = float(np.mean([bool(row["success"]) for row in python_episode_rows]))
                violation_rate = float(np.mean([bool(row["violation"]) for row in python_episode_rows]))
            else:
                _validate_episode_parity(scala_episode_rows, python_episode_rows)
                environment_parity = "verified episode-by-episode"
                evaluation_seconds = float(result["evaluation_seconds"])
                evaluation_calls = int(result["evaluation_model_calls"])
                mean_cost = float(result["mean_cost"])
                success_rate = float(result["success_rate"])
                violation_rate = float(result["violation_rate"])
            training_seconds = float(result["training_seconds"])
            training_calls = int(result["training_model_calls"])
            diagnostics_getter = getattr(python_env, "evaluation_action_diagnostics", None)
            action_diagnostics = dict(diagnostics_getter()) if callable(diagnostics_getter) else {}
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": "SymPar-Q-Upstream",
                    "seed": seed,
                    "seconds": partition_seconds
                    + training_seconds
                    + evaluation_seconds,
                    "preparation_seconds": partition_seconds + training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "training_seconds": training_seconds,
                    "partition_seconds": partition_seconds,
                    "scala_process_seconds": scala_process_seconds,
                    "training_episodes": args.training_episodes,
                    "evaluation_episodes": args.evaluation_episodes,
                    "iterations": args.training_episodes,
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "epsilon0": args.epsilon0,
                    "epsilon_decay_factor": args.epsilon_decay_factor,
                    "min_exploration": args.min_exploration,
                    "reward_mode": args.reward_mode,
                    "success_bonus": args.success_bonus,
                    "potential_scale": args.potential_scale,
                    "partition_variant": args.partition_variant,
                    "training_initialization": args.training_initialization,
                    "mean_cost": mean_cost,
                    "success_rate": success_rate,
                    "violation_rate": violation_rate,
                    "blocks": len(blocks),
                    "model_calls": training_calls,
                    "model_call_accounting": "training only; excludes actual evaluation transitions",
                    "training_model_calls": training_calls,
                    "evaluation_model_calls": evaluation_calls,
                    "learned_policy_entries": int(result["learned_policy_entries"]),
                    "defaulted_blocks": int(result["defaulted_blocks"]),
                    "defaulted_block_ids": " ".join(
                        str(block_id) for block_id in result["defaulted_block_ids"]
                    ),
                    "status": "ok",
                    "source_artifact": "zenodo.14620119",
                    "partition_source": (
                        "SDTA supplied white-box partition"
                        if args.partition_variant == "sdta"
                        else "SDTA supplied white-box partition + diagnostic control refinement"
                    ),
                    "environment_parity": environment_parity,
                    "transition_regime": evaluation_env.transition_regime(),
                    "noise_model": action_diagnostics.get("noise_model", ""),
                    "noise_scale": evaluation_env.transition_noise_scale(),
                    "action_sample_count": action_diagnostics.get("action_sample_count", 0),
                    "action_perturbation_count": action_diagnostics.get("action_perturbation_count", 0),
                    "action_perturbation_rate": action_diagnostics.get("action_perturbation_rate", ""),
                    "mean_abs_action_deviation": action_diagnostics.get("mean_abs_action_deviation", ""),
                    "mean_intended_action": action_diagnostics.get("mean_intended_action", ""),
                    "mean_executed_action": action_diagnostics.get("mean_executed_action", ""),
                    "evaluation_owner": (
                        "Python unified stochastic protocol"
                        if args.stochastic_slip_probability > 0.0
                        else "Scala/Python parity"
                    ),
                }
            )
        _write_results(args.out_dir / "results.csv", rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official SymPar Scala Q-learning backend over supplied "
            "SDTA white-box partitions."
        )
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=DEFAULT_UPSTREAM_ROOT,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=DEFAULT_BENCHMARKS,
        default=list(DEFAULT_BENCHMARKS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
    )
    parser.add_argument("--training-episodes", type=int, default=20_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon0", type=float, default=1.0)
    parser.add_argument("--epsilon-decay-factor", type=float, default=0.99)
    parser.add_argument("--min-exploration", type=float, default=1e-5)
    parser.add_argument(
        "--reward-mode",
        choices=("matched-cost",),
        default="matched-cost",
    )
    parser.add_argument("--success-bonus", type=float, default=1_000.0)
    parser.add_argument("--potential-scale", type=float, default=10.0)
    parser.add_argument(
        "--partition-variant",
        choices=("control-refined",),
        default="control-refined",
    )
    parser.add_argument(
        "--training-initialization",
        choices=("mixed-witness",),
        default="mixed-witness",
    )
    parser.add_argument(
        "--stochastic-slip-probability",
        type=float,
        default=0.1,
        help=(
            "Replace the intended finite action with another action during "
            "upstream Q-learning. Use 0.10 for the stochastic main protocol."
        ),
    )
    parser.add_argument("--scala-cli", default="scala-cli")
    parser.add_argument("--scala-version", default="3.3.0")
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "sympar",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.stochastic_slip_probability <= 1.0:
        raise ValueError("--stochastic-slip-probability must lie in [0, 1]")
    args.benchmarks = tuple(args.benchmarks)
    args.seeds = tuple(args.seeds)
    rows = run(args)
    print(
        f"Completed {len(rows)} upstream SymPar-Q rows in {args.out_dir.resolve()}."
    )


if __name__ == "__main__":
    main()
