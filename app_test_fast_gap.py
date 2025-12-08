"""
Fast GRPO vs GVPO Benchmark
===========================

This script builds a reproducible MAPF benchmark that compares the original GRPO
and the new GVPO allocator policies under the same conditions. The benchmark
accepts user-specified map / scenario files and scales naturally to large
instances (80x80, 100x100+, 200+ agents) so that congestion differences appear
without artificial tweaks. Both algorithms share the exact same architecture,
hyperparameters, and training loop length; their internal logic is the only
difference.

Key improvements over the previous script:
    * Mandatory CLI arguments for map, scenario, agent count, and task count.
      The run fails fast if files are missing so that we always benchmark real
      MAPF instances instead of a synthetic 20x20 toy grid.
    * Environment difficulty now comes from the chosen data: larger maps and
      more agents automatically induce more collisions and makespan pressure.
    * Full reproducibility: all randomness is seeded per episode so GRPO and
      GVPO see identical initial states. We support multiple benchmark episodes
      to capture variance.
    * Detailed logging: each episode prints success_rate, avg makespan,
      best_cost, and inference_time for GRPO and GVPO, making it easy to inspect
      how they diverge. A final table summarizes averaged metrics and deltas.
    * Modular structure ready for Python 3.11 / NumPy 1.26.4 / torch >= 2.0.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", enqueue=False, backtrace=False, diagnose=False)

from pycam import get_grid, get_scenario
from pycam.allo_util_RL import RLAllocator
from pycam.mapf_utils import Config
from pycam.planner_base import PlannerRequest, PlannerResult
from pycam.planner_factory import PlannerFactory


# --------------------------------------------------------------------------- #
# Helper data structures
# --------------------------------------------------------------------------- #


@dataclass
class Environment:
    map_path: Path
    scen_path: Path
    grid: np.ndarray
    starts: Config
    goals: Config
    agent_dict: Dict[int, Tuple[int, int]]
    task_dict: Dict[int, List]
    num_agents: int
    num_tasks: int
    init_plan: Dict[str, List[str]]
    optimize_funcs: List[Callable]
    reward_fn: Callable[[Dict[str, List[str]]], float]


@dataclass
class EpisodeResult:
    success_rate: float
    avg_makespan: float
    best_cost: float
    inference_time: float


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO vs GVPO MAPF benchmark")
    parser.add_argument("--map", required=True, type=Path, help="Path to MAPF map file (*.map)")
    parser.add_argument("--scenario", required=True, type=Path, help="Path to MAPF scenario file (*.scen)")
    parser.add_argument("--agents", required=True, type=int, help="Number of agents to load from the scenario")
    parser.add_argument("--tasks", required=True, type=int, help="Number of pickup/dropoff tasks to load")
    parser.add_argument("--episodes", type=int, default=1, help="Benchmark repetitions per algorithm")
    parser.add_argument("--max-episodes", type=int, default=120, help="RL training steps (same for both algos)")
    parser.add_argument(
        "--train-time-budget-s",
        type=float,
        default=0.0,
        help="Wall-clock training budget per allocator (seconds). 0 disables the budget.",
    )
    parser.add_argument("--batch-size", type=int, default=192, help="Batch size for PPO-style replay")
    parser.add_argument("--generations", type=int, default=6, help="Population size for GRPO/GVPO sampling")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--log-interval", type=int, default=1, help="Episode logging frequency")
    parser.add_argument(
        "--planner",
        type=str,
        choices=PlannerFactory.choices(),
        default="lacam",
        help="Planner backend for MAPF solving (default: lacam)",
    )
    parser.add_argument(
        "--planner-time-limit-ms",
        type=int,
        default=5000,
        help="Time limit for each planner call in milliseconds",
    )
    parser.add_argument(
        "--planner-neighbours",
        type=int,
        default=5,
        help="Neighbourhood size passed to the LNS2 planner",
    )
    parser.add_argument(
        "--planner-eval-every",
        type=int,
        default=0,
        help="Run the MAPF planner every N episodes (0 = only run after the final episode)",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "hard"],
        default="default",
        help="Convenience preset overriding agents/tasks/max_episodes for harder regimes",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_initial_plan(num_agents: int, num_tasks: int) -> Dict[str, List[str]]:
    plan = {f"a{i}": [] for i in range(num_agents)}
    for task_idx in range(num_tasks):
        plan[f"a{task_idx % num_agents}"].append(f"t{task_idx}")
    return plan


def compute_single_agent_cost(agent_idx: int, tasks: List[str], agent_dict, task_dict) -> float:
    current = agent_dict.get(agent_idx)
    if current is None:
        return 0.0
    total = 0.0
    for task_name in tasks:
        task_idx = int(task_name[1:])
        pickup, dropoff, _ = task_dict.get(task_idx, (current, current, 0))
        total += manhattan(current, pickup)
        total += manhattan(pickup, dropoff)
        current = dropoff
    return total


def compute_agent_cost_map(plan, agent_dict, task_dict) -> Dict[str, float]:
    cost_map = {}
    for agent_name, tasks in plan.items():
        agent_idx = int(agent_name[1:])
        cost_map[agent_name] = compute_single_agent_cost(agent_idx, tasks, agent_dict, task_dict)
    return cost_map


def build_optimize_funcs(num_agents: int, num_tasks: int, agent_dict, task_dict) -> List[Callable]:
    def move_task(plan):
        mutated = copy.deepcopy(plan)
        src = f"a{random.randrange(num_agents)}"
        dst = f"a{random.randrange(num_agents)}"
        if mutated[src]:
            task = mutated[src].pop(random.randrange(len(mutated[src])))
            mutated[dst].insert(random.randrange(len(mutated[dst]) + 1), task)
        return mutated

    def swap_tasks(plan):
        mutated = copy.deepcopy(plan)
        if num_agents < 2:
            return mutated
        a_idx, b_idx = random.sample(range(num_agents), k=2)
        agent_a, agent_b = f"a{a_idx}", f"a{b_idx}"
        if mutated[agent_a] and mutated[agent_b]:
            i = random.randrange(len(mutated[agent_a]))
            j = random.randrange(len(mutated[agent_b]))
            mutated[agent_a][i], mutated[agent_b][j] = mutated[agent_b][j], mutated[agent_a][i]
        return mutated

    def reinsert_block(plan):
        mutated = copy.deepcopy(plan)
        donors = [agent for agent, tasks in mutated.items() if tasks]
        if not donors:
            return mutated
        src = random.choice(donors)
        src_tasks = mutated[src]
        block_len = random.randint(1, min(3, len(src_tasks)))
        start_idx = random.randrange(0, len(src_tasks) - block_len + 1)
        block = src_tasks[start_idx : start_idx + block_len]
        del src_tasks[start_idx : start_idx + block_len]
        dst = random.choice(list(mutated.keys()))
        insert_idx = random.randrange(len(mutated[dst]) + 1)
        for offset, task in enumerate(block):
            mutated[dst].insert(insert_idx + offset, task)
        return mutated

    def two_opt_within_agent(plan):
        mutated = copy.deepcopy(plan)
        candidates = [agent for agent, tasks in mutated.items() if len(tasks) >= 2]
        if not candidates:
            return mutated
        agent_name = random.choice(candidates)
        tasks = mutated[agent_name]
        i, j = sorted(random.sample(range(len(tasks)), 2))
        before_cost = compute_single_agent_cost(int(agent_name[1:]), tasks, agent_dict, task_dict)
        new_order = tasks[:i] + list(reversed(tasks[i : j + 1])) + tasks[j + 1 :]
        after_cost = compute_single_agent_cost(int(agent_name[1:]), new_order, agent_dict, task_dict)
        if after_cost < before_cost:
            mutated[agent_name] = new_order
        return mutated

    def balance_heavy_to_light(plan):
        mutated = copy.deepcopy(plan)
        cost_map = compute_agent_cost_map(mutated, agent_dict, task_dict)
        if not cost_map:
            return mutated
        heavy = max(cost_map.items(), key=lambda kv: kv[1])[0]
        light = min(cost_map.items(), key=lambda kv: kv[1])[0]
        if heavy == light or not mutated[heavy]:
            return mutated
        task_idx = random.randrange(len(mutated[heavy]))
        task = mutated[heavy].pop(task_idx)
        insert_idx = random.randrange(len(mutated[light]) + 1)
        mutated[light].insert(insert_idx, task)
        return mutated

    return [move_task, swap_tasks, reinsert_block, two_opt_within_agent, balance_heavy_to_light]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_total_cost(plan: Dict[str, List[str]], agent_dict, task_dict) -> float:
    total = 0.0
    for agent_name, tasks in plan.items():
        agent_idx = int(agent_name[1:])
        current = agent_dict.get(agent_idx)
        if current is None:
            continue
        for task_name in tasks:
            task_idx = int(task_name[1:])
            pickup, dropoff, _ = task_dict.get(task_idx, (current, current, 0))
            total += manhattan(current, pickup)
            total += manhattan(pickup, dropoff)
            current = dropoff
    return total


def compute_agent_costs(plan, agent_dict, task_dict) -> List[float]:
    costs = []
    for agent_name, tasks in plan.items():
        agent_idx = int(agent_name[1:])
        current = agent_dict.get(agent_idx)
        if current is None:
            continue
        agent_cost = 0.0
        for task_name in tasks:
            task_idx = int(task_name[1:])
            pickup, dropoff, _ = task_dict.get(task_idx, (current, current, 0))
            agent_cost += manhattan(current, pickup)
            agent_cost += manhattan(pickup, dropoff)
            current = dropoff
        costs.append(agent_cost)
    return costs


def compute_success(plan: Dict[str, List[str]], num_tasks: int) -> float:
    assigned = [task for tasks in plan.values() for task in tasks]
    return min(1.0, len(set(assigned)) / max(1, num_tasks))


def build_reward_fn(agent_dict, task_dict) -> Callable:
    def reward(plan):
        return -compute_total_cost(plan, agent_dict, task_dict)

    return reward


def build_planner_request(env: Environment, time_limit_ms: int, seed: int) -> PlannerRequest:
    starts_cfg = Config(list(env.starts.positions[: env.num_agents]))
    goals_cfg = Config(list(env.goals.positions[: env.num_agents]))
    return PlannerRequest(
        grid=env.grid,
        starts=starts_cfg,
        goals=goals_cfg,
        time_limit_ms=time_limit_ms,
        seed=seed,
        verbose=1,
    )


def load_environment(args: argparse.Namespace) -> Environment:
    if not args.map.exists():
        raise FileNotFoundError(f"Map file not found: {args.map}")
    if not args.scenario.exists():
        raise FileNotFoundError(f"Scenario file not found: {args.scenario}")

    grid, _, _ = get_grid(args.map)
    starts, goals, _, agent_dict, task_dict = get_scenario(args.scenario, args.agents, args.tasks)

    if len(agent_dict) < args.agents:
        raise ValueError(f"Scenario only provided {len(agent_dict)} agents (< {args.agents})")
    if len(task_dict) < args.tasks:
        raise ValueError(f"Scenario only provided {len(task_dict)} tasks (< {args.tasks})")

    agent_map = {int(k): v for k, v in agent_dict.items()}
    task_map = {int(k): v for k, v in task_dict.items()}
    env = Environment(
        map_path=args.map,
        scen_path=args.scenario,
        grid=grid,
        starts=starts,
        goals=goals,
        agent_dict=agent_map,
        task_dict=task_map,
        num_agents=args.agents,
        num_tasks=args.tasks,
        init_plan=build_initial_plan(args.agents, args.tasks),
        optimize_funcs=build_optimize_funcs(args.agents, args.tasks, agent_map, task_map),
        reward_fn=build_reward_fn(agent_map, task_map),
    )
    return env


def should_evaluate_planner(episode_idx: int, total_episodes: int, eval_every: int) -> bool:
    if total_episodes <= 0:
        return False
    if eval_every <= 0:
        return episode_idx == total_episodes - 1
    if (episode_idx + 1) % eval_every == 0:
        return True
    return episode_idx == total_episodes - 1


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #


def run_allocator(
    algo_name: str,
    env: Environment,
    shared_hparams: Dict,
    planner_request: Optional[PlannerRequest],
) -> Tuple[Dict[str, List[str]], float, float, Optional[PlannerResult]]:
    allocator = RLAllocator(
        env.num_agents,
        env.num_tasks,
        env.optimize_funcs,
        env.reward_fn,
        algo_type=algo_name,
        max_episodes=shared_hparams["max_episodes"],
        batch_size=shared_hparams["batch_size"],
        num_generations=shared_hparams["num_generations"],
        learning_rate=shared_hparams["learning_rate"],
        planner_type=shared_hparams["planner_type"],
        planner_kwargs=shared_hparams["planner_kwargs"],
    )
    allocator.set_train_time_budget(shared_hparams.get("train_time_budget_s"))
    start = time.time()
    plan, best_cost = allocator.train(copy.deepcopy(env.init_plan))
    runtime = time.time() - start
    planner_result = None
    if planner_request is not None:
        try:
            planner_result = allocator.solve_mapf(planner_request)
        except Exception as exc:
            logger.warning(
                "[%s] Planner '%s' failed: %s",
                algo_name.upper(),
                shared_hparams["planner_type"],
                exc,
            )
    return plan, best_cost, runtime, planner_result


def summarize_plan(plan, env, evaluated_cost, inference_time) -> EpisodeResult:
    success = compute_success(plan, env.num_tasks)
    agent_costs = compute_agent_costs(plan, env.agent_dict, env.task_dict)
    avg_makespan = float(np.mean(agent_costs)) if agent_costs else 0.0
    return EpisodeResult(
        success_rate=success,
        avg_makespan=avg_makespan,
        best_cost=float(evaluated_cost),
        inference_time=float(inference_time),
    )


def run_episode(
    algo_name: str,
    env: Environment,
    shared_hparams: Dict,
    episode_idx: int,
    total_episodes: int,
    base_seed: int,
    should_eval_planner: bool,
) -> EpisodeResult:
    episode_seed = base_seed + episode_idx
    seed_everything(episode_seed)
    logger.info("[%s][Episode %d/%d] Seed=%d", algo_name.upper(), episode_idx + 1, total_episodes, episode_seed)
    planner_request = None
    if should_eval_planner:
        planner_request = build_planner_request(env, shared_hparams["planner_time_limit_ms"], episode_seed)
    plan, reported_cost, runtime, planner_result = run_allocator(algo_name, env, shared_hparams, planner_request)
    manhattan_cost = compute_total_cost(plan, env.agent_dict, env.task_dict)
    result = summarize_plan(plan, env, manhattan_cost, runtime)
    logger.info(
        "[%s][Episode %d/%d] success_rate=%.3f avg_makespan=%.1f best_cost=%.1f inference_time=%.2fs",
        algo_name.upper(),
        episode_idx + 1,
        total_episodes,
        result.success_rate,
        result.avg_makespan,
        result.best_cost,
        result.inference_time,
    )
    logger.info(
        "[%s][Episode %d/%d] allocator_reported_cost=%.1f manhattan_cost=%.1f",
        algo_name.upper(),
        episode_idx + 1,
        total_episodes,
        reported_cost,
        manhattan_cost,
    )
    if planner_result and planner_result.cost is not None:
        logger.info(
            "[%s][Episode %d/%d] planner=%s cost=%.1f runtime=%.2fs",
            algo_name.upper(),
            episode_idx + 1,
            total_episodes,
            shared_hparams["planner_type"],
            planner_result.cost,
            planner_result.runtime_s or 0.0,
        )
    return result


def aggregate(results: List[EpisodeResult]) -> EpisodeResult:
    if not results:
        return EpisodeResult(0.0, 0.0, 0.0, 0.0)
    success = np.mean([r.success_rate for r in results])
    makespan = np.mean([r.avg_makespan for r in results])
    cost = np.mean([r.best_cost for r in results])
    runtime = np.mean([r.inference_time for r in results])
    return EpisodeResult(float(success), float(makespan), float(cost), float(runtime))


def print_table(grpo_stats: EpisodeResult, gvpo_stats: EpisodeResult):
    header = f"{'metric':<20}{'GRPO':>12}{'GVPO':>12}{'delta (GVPO-GRPO)':>22}{'rel %':>10}"
    print("\n" + header)
    print("-" * len(header))
    rows = [
        ("success_rate", grpo_stats.success_rate, gvpo_stats.success_rate),
        ("avg_makespan", grpo_stats.avg_makespan, gvpo_stats.avg_makespan),
        ("best_cost", grpo_stats.best_cost, gvpo_stats.best_cost),
        ("inference_time", grpo_stats.inference_time, gvpo_stats.inference_time),
    ]
    for key, g_val, v_val in rows:
        delta = v_val - g_val
        rel = 0.0 if g_val == 0 else (delta / abs(g_val)) * 100.0
        print(f"{key:<20}{g_val:>12.4f}{v_val:>12.4f}{delta:>22.4f}{rel:>10.2f}")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    if args.preset == "hard":
        hard_agents = max(args.agents, 320)
        hard_tasks = max(args.tasks, 320)
        hard_episodes = max(args.max_episodes, 200)
        if (hard_agents, hard_tasks, hard_episodes) != (args.agents, args.tasks, args.max_episodes):
            logger.info(
                "Applying hard preset overrides: agents %d->%d tasks %d->%d max_episodes %d->%d",
                args.agents,
                hard_agents,
                args.tasks,
                hard_tasks,
                args.max_episodes,
                hard_episodes,
            )
        args.agents, args.tasks, args.max_episodes = hard_agents, hard_tasks, hard_episodes
    logger.info("Loading environment: map=%s scenario=%s", args.map, args.scenario)
    env = load_environment(args)

    planner_type = args.planner.lower()
    planner_kwargs: Dict[str, int] = {}
    if planner_type == "lns2":
        planner_kwargs["num_neighbours"] = args.planner_neighbours

    shared_hparams = {
        "max_episodes": args.max_episodes,
        "batch_size": args.batch_size,
        "num_generations": args.generations,
        "learning_rate": 1e-4,
        "planner_type": planner_type,
        "planner_kwargs": planner_kwargs,
        "planner_time_limit_ms": args.planner_time_limit_ms,
        "train_time_budget_s": max(0.0, float(args.train_time_budget_s)),
    }

    grpo_results, gvpo_results = [], []
    for episode_idx in range(args.episodes):
        eval_flag = should_evaluate_planner(episode_idx, args.episodes, args.planner_eval_every)
        grpo_results.append(
            run_episode("grpo", env, shared_hparams, episode_idx, args.episodes, args.seed, eval_flag)
        )
        gvpo_results.append(
            run_episode("gvpo", env, shared_hparams, episode_idx, args.episodes, args.seed + 10_000, eval_flag)
        )

    grpo_stats = aggregate(grpo_results)
    gvpo_stats = aggregate(gvpo_results)
    print_table(grpo_stats, gvpo_stats)


if __name__ == "__main__":
    main()
