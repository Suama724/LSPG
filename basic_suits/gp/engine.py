import copy
import os
import time
from typing import Callable, Optional

import numpy as np
import ray

from .evaluator import FitnessEvaluator, DEFAULT_FAILURE_PENALTY
from .evolution import crossover, subtree_mutation, hoist_mutation, point_mutation
from .structure import ExpressionTree
from .utils import check_random_state


@ray.remote(num_cpus=1)
def _evaluate_individual_remote(individual, evaluator):
    evaluator.calculate_fitness(individual)
    return individual


class EvolutionEngine:
    def __init__(
        self,
        population_size: int=1000,
        generations: int=10,
        n_jobs: int=1,
        random_state: np.random.Generator=None,
        verbose: int=0,
        tournament_size: int=25,
        init_depth=(3, 6),
        mutate_depth=(0, 2),
        p_crossover: float=0.7,
        p_subtree_mutation: float=0.15,
        p_hoist_mutation: float=0.05,
        p_point_mutation: float=0.05,
        p_point_replace: float=0.1,
        elitism: int=1,
        reallocate_interval: int=10,
        reallocate_delta: int=2,
        pmin=10,
        use_ray_min_population=16,
        w_len: float=0.03,
        w_depth: float=0.03,
        w_dim: float=0.0,
        w_symbol_cost: float=0.01,
        failure_penalty=DEFAULT_FAILURE_PENALTY,
        device="cpu",
    ):
        self.population_size = population_size
        self.generations = generations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tournament_size = tournament_size
        self.init_depth = init_depth
        self.mutate_depth = mutate_depth
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.elitism = elitism # number of elites to keep in the population
        self.reallocate_interval = reallocate_interval # interval at which to reallocate the population
        self.reallocate_delta = reallocate_delta # delta to add to the best task
        self.pmin = pmin # minimum population size for each task
        self.use_ray_min_population = use_ray_min_population # minimum population size to use ray
        self.w_len = w_len
        self.w_depth = w_depth
        self.w_dim = w_dim
        self.w_symbol_cost = w_symbol_cost
        self.failure_penalty = failure_penalty
        self.device = device

        self._method_probs = np.cumsum(
            [
                self.p_crossover,
                self.p_subtree_mutation,
                self.p_hoist_mutation,
                self.p_point_mutation,
            ]
        )
        if self._method_probs[-1] > 1.0:
            raise ValueError("Mutation/crossover probabilities must sum <= 1.0")

    def _build_tasks(self, dataset_dict, target_coords):
        items = list(dataset_dict.items())
        n_tasks = len(items)
        tasks = []

        for i, (task_key, task_data) in enumerate(items):
            if isinstance(task_data, (tuple, list)) and len(task_data) >= 1:
                X = task_data[0]
            else:
                X = task_data
            X = np.asarray(X, dtype=np.float64)

            target_coord = None
            if isinstance(target_coords, dict):
                target_coord = target_coords.get(task_key, None)
            elif isinstance(target_coords, (list, tuple)):
                if len(target_coords) == n_tasks:
                    target_coord = target_coords[i]
                elif len(target_coords) > 0:
                    target_coord = target_coords[0]
            elif target_coords is not None:
                arr = np.asarray(target_coords)
                target_coord = arr if arr.ndim == 1 else arr[i]

            if target_coord is not None:
                target_coord = np.asarray(target_coord, dtype=np.float64).ravel()

            tasks.append(
                {
                    "X": X,
                    "target_coord": target_coord,
                    "target_dim": int(X.shape[1]),
                }
            )
        return tasks

    def _default_alloc(self, n_tasks):
        if n_tasks <= 0:
            return []
        base = self.population_size // n_tasks
        alloc = [base] * n_tasks
        for i in range(self.population_size - base * n_tasks):
            alloc[i % n_tasks] += 1
        return alloc

    def _task_best_list(self, population, n_tasks):
        out = [self.failure_penalty] * n_tasks
        for ind in population:
            if not ind.raw_per_task:
                continue
            for t in range(n_tasks):
                out[t] = min(out[t], float(ind.raw_per_task[t]))
        return out

    def _fallback_mark_failure(self, individual, n_tasks):
        individual.raw_per_task = [self.failure_penalty] * n_tasks
        individual.raw_dist_per_task = [self.failure_penalty] * n_tasks
        individual.penalty_per_task = [0.0] * n_tasks
        individual.best_task_id = 0
        individual.skill_factor = 0
        individual.raw_dist = self.failure_penalty
        individual.scaler_fitness = self.failure_penalty
        individual.fitness = self.failure_penalty
        return individual

    def _collect_task_snapshots(self, population, n_tasks):
        snapshots = []
        for t in range(n_tasks):
            best_ind = None
            best_total = self.failure_penalty
            for ind in population:
                if not getattr(ind, "raw_per_task", None):
                    continue
                if t >= len(ind.raw_per_task):
                    continue
                v = float(ind.raw_per_task[t])
                if v < best_total:
                    best_total = v
                    best_ind = ind
            if best_ind is None:
                snapshots.append(
                    {
                        "task_id": t,
                        "best_total": float(self.failure_penalty),
                        "best_raw_dist": float(self.failure_penalty),
                        "best_penalty": 0.0,
                        "best_program": None,
                    }
                )
            else:
                raw_dist_list = getattr(best_ind, "raw_dist_per_task", [])
                penalty_list = getattr(best_ind, "penalty_per_task", [])
                best_raw_dist = (
                    float(raw_dist_list[t])
                    if t < len(raw_dist_list)
                    else float(best_total)
                )
                best_penalty = (
                    float(penalty_list[t])
                    if t < len(penalty_list)
                    else float(best_total - best_raw_dist)
                )
                snapshots.append(
                    {
                        "task_id": t,
                        "best_total": float(best_total),
                        "best_raw_dist": float(best_raw_dist),
                        "best_penalty": float(best_penalty),
                        "best_program": str(best_ind),
                    }
                )
        return snapshots

    @staticmethod
    def _format_generation_log_block(rec, elapsed_s):
        task_best = rec.get("task_best_list", [])
        pop_alloc = rec.get("pop_alloc", [])
        task_best_text = ", ".join(f"{float(x):.6g}" for x in task_best) if task_best else "-"
        pop_alloc_text = ", ".join(str(int(x)) for x in pop_alloc) if pop_alloc else "-"
        sep = "=" * 88
        lines = [
            sep,
            f"[Gen] {int(rec.get('gen', -1))}",
            f"[TimeCostSec] {float(elapsed_s):.4f}",
            f"[GlobalBestFitness] {float(rec.get('best_fitness', np.nan)):.12g}",
            f"[GlobalMeanFitness] {float(rec.get('mean_fitness', np.nan)):.12g}",
            f"[BestTaskId] {int(rec.get('best_task_id', -1))}",
            f"[TaskBestList] {task_best_text}",
            f"[PopulationAllocation] {pop_alloc_text}",
            sep,
            "",
        ]
        return "\n".join(lines)

    def _evaluate_population(self, population, evaluator):
        n_tasks = len(evaluator.tasks)
        use_ray = (
            self.n_jobs != 1
            and len(population) >= self.use_ray_min_population
            and ray.is_initialized()
        )
        if not use_ray:
            for i, ind in enumerate(population):
                try:
                    evaluator.calculate_fitness(ind)
                except Exception:
                    population[i] = self._fallback_mark_failure(ind, n_tasks)
            return population

        evaluator_ref = ray.put(evaluator)
        futures = [_evaluate_individual_remote.remote(ind, evaluator_ref) for ind in population]
        evaluated = []
        for i, fut in enumerate(futures):
            try:
                evaluated.append(ray.get(fut))
            except Exception:
                evaluated.append(self._fallback_mark_failure(population[i], n_tasks))
        return evaluated

    def _build_task_pools(self, population, n_tasks):
        pools = {t: [] for t in range(n_tasks)}
        for idx, ind in enumerate(population):
            t = int(ind.best_task_id) if ind.best_task_id is not None else 0
            t = max(0, min(n_tasks - 1, t))
            pools[t].append(idx)
        return pools

    def _sample_task_by_alloc(self, alloc, random_state):
        if not alloc:
            return 0
        weights = np.asarray(alloc, dtype=np.float64)
        if np.sum(weights) <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
        probs = weights / np.sum(weights)
        return int(random_state.choice(len(alloc), p=probs))

    def _tournament_select(self, population, pools, task_id, random_state):
        candidate_pool = pools.get(task_id, [])
        if len(candidate_pool) < self.tournament_size:
            candidate_pool = list(range(len(population)))
        contenders = random_state.choice(candidate_pool, size=self.tournament_size, replace=True)
        best_idx = min(contenders, key=lambda i: population[i].scaler_fitness)
        return population[int(best_idx)]

    def _make_offspring(self, population, pools, task_id, random_state, n_feats):
        parent = self._tournament_select(population, pools, task_id, random_state)
        r = random_state.random()
        if r < self._method_probs[0]:
            donor = self._tournament_select(population, pools, task_id, random_state)
            return crossover(donor, parent, random_state)
        if r < self._method_probs[1]:
            return subtree_mutation(parent, random_state, n_feats, init_depth=self.mutate_depth)
        if r < self._method_probs[2]:
            return hoist_mutation(parent, random_state)
        if r < self._method_probs[3]:
            return point_mutation(parent, random_state, self.p_point_replace, n_feats)
        return copy.deepcopy(parent)

    def _reallocate(self, alloc, task_best_list):
        if not alloc or len(alloc) <= 1:
            return alloc
        new_alloc = list(alloc)
        best_id = int(np.argmin(task_best_list))

        # Increase best task by delta.
        new_alloc[best_id] += self.reallocate_delta
        need_reduce = self.reallocate_delta
        others = [i for i in range(len(new_alloc)) if i != best_id]

        # Reduce from other tasks while respecting Pmin.
        while need_reduce > 0 and others:
            progressed = False
            for i in others:
                if need_reduce <= 0:
                    break
                if new_alloc[i] > self.pmin:
                    new_alloc[i] -= 1
                    need_reduce -= 1
                    progressed = True
            if not progressed:
                break

        # Keep total unchanged.
        diff = self.population_size - sum(new_alloc)
        if diff > 0:
            new_alloc[best_id] += diff
        elif diff < 0:
            for i in sorted(range(len(new_alloc)), key=lambda x: new_alloc[x], reverse=True):
                while diff < 0 and new_alloc[i] > self.pmin:
                    new_alloc[i] -= 1
                    diff += 1
                if diff == 0:
                    break
        return new_alloc

    def fit(
        self,
        dataset_dict,
        model,
        scaler,
        target_coords,
        n_feats,
        log_path=None,
        on_generation_end: Optional[Callable[[int, dict, float], None]] = None,
    ):
        random_state = check_random_state(self.random_state)
        tasks = self._build_tasks(dataset_dict, target_coords)
        if not tasks:
            return None, []
        task_keys = list(dataset_dict.keys())
        n_tasks = len(tasks)

        evaluator = FitnessEvaluator(
            tasks=tasks,
            model=model,
            scaler=scaler,
            device=self.device,
            random_state=int(random_state.integers(0, 1_000_000)),
            w_len=self.w_len,
            w_depth=self.w_depth,
            w_dim=self.w_dim,
            w_symbol_cost=self.w_symbol_cost,
            failure_penalty=self.failure_penalty,
        )

        population = [
            ExpressionTree.create_random_tree(
                random_state,
                total_dim=n_feats,
                init_depth=self.init_depth,
                method="half and half",
            )
            for _ in range(self.population_size)
        ]
        alloc = self._default_alloc(len(tasks))
        history = []
        global_best = None
        global_best_fitness = self.failure_penalty
        best_per_task_program = [None] * n_tasks
        best_per_task_fitness = [self.failure_penalty] * n_tasks

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")

        for gen in range(self.generations):
            t0 = time.time()
            population = self._evaluate_population(population, evaluator)
            population.sort(key=lambda ind: ind.scaler_fitness)
            best = population[0]
            mean_fitness = float(np.mean([ind.scaler_fitness for ind in population]))
            task_best_list = self._task_best_list(population, n_tasks)
            task_snapshots = self._collect_task_snapshots(population, n_tasks)

            for ind in population:
                if not ind.raw_per_task:
                    continue
                for t in range(min(n_tasks, len(ind.raw_per_task))):
                    v = float(ind.raw_per_task[t])
                    if v < best_per_task_fitness[t]:
                        best_per_task_fitness[t] = v
                        best_per_task_program[t] = copy.deepcopy(ind)

            if best.scaler_fitness < global_best_fitness:
                global_best = copy.deepcopy(best)
                global_best_fitness = float(best.scaler_fitness)

            rec = {
                "gen": gen,
                "best_fitness": float(best.scaler_fitness),
                "mean_fitness": mean_fitness,
                "task_best_list": [float(x) for x in task_best_list],
                "pop_alloc": list(alloc),
                "best_task_id": int(best.best_task_id),
                "best_program": copy.deepcopy(best),
                "task_snapshots": task_snapshots,
            }
            history.append(rec)
            elapsed_s = time.time() - t0

            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(self._format_generation_log_block(rec, elapsed_s))

            if on_generation_end is not None:
                on_generation_end(gen, rec, elapsed_s)

            if self.verbose:
                print(
                    f"[Gen {gen:03d}] best={rec['best_fitness']:.6f} "
                    f"mean={rec['mean_fitness']:.6f} task={rec['best_task_id']}"
                )

            if gen == self.generations - 1:
                break

            if self.reallocate_interval > 0 and (gen + 1) % self.reallocate_interval == 0:
                alloc = self._reallocate(alloc, task_best_list)

            pools = self._build_task_pools(population, n_tasks)
            elites = [copy.deepcopy(population[i]) for i in range(min(self.elitism, len(population)))]
            offspring = []
            n_offspring = self.population_size - len(elites)
            for _ in range(n_offspring):
                task_id = self._sample_task_by_alloc(alloc, random_state)
                child = self._make_offspring(population, pools, task_id, random_state, n_feats)
                offspring.append(child)
            population = elites + offspring

        if n_tasks == 1:
            return global_best, history
        best_programs_by_task = {
            task_keys[i]: best_per_task_program[i] for i in range(n_tasks)
        }
        return best_programs_by_task, history
