import time
import itertools
import numpy as np
import ray
from tqdm import tqdm
from copy import deepcopy

from .structure import ExpressionTree
from .evolution import EvolutionOps
from .evaluator import FitnessEvaluator
from .utils import check_random_state, _partition_estimators

@ray.remote
def _evolve_and_evaluate_batch(
        n_programs: int,
        parents,
        random_seed,
        dataset_dict,
        model,
        scaler,
        target_coords,
        n_features,
        init_depth,
        method_probs,
        tournament_size,
        p_point_replace,
        feature_names,
        parsimony_coefficient
    ):
    random_state = check_random_state(random_seed)
    evaluator = FitnessEvaluator(dataset_dict, model, scaler, target_coords)

    programs = []

    performence_stats = evaluator.perfomence_stats
    count = performence_stats['cnt']
    if count > 0:
        avg_exec = (performence_stats['execute_time'] / count) * 1000  # ms
        avg_ela = (performence_stats['ela_time'] / count) * 1000    
        avg_model = (performence_stats['encode_time'] / count) * 1000 
        
    '''
    这里print可能不显示
        print(f"[Worker Stats] Processed {count} individuals. "
              f"Avg Times -> Exec: {avg_exec:.2f}ms | "
              f"ELA: {avg_ela:.2f}ms | "
              f"Model: {avg_model:.2f}ms")    
    '''  

    def tournament():
        if parents is None: 
            return None
        contenders_indices = random_state.integers(0, len(parents), tournament_size)
        contenders = [parents[i] for i in contenders_indices]
        winner = min(contenders, key=lambda x: x.raw_fitness_ if x.fitness_ is not None else x.raw_fitness_)
        return winner        

    evolution_ops_time_total = 0.0

    for _ in range(n_programs):
        child = None
        t_evolution_start = time.perf_counter()

        if parents is None:
            child = ExpressionTree.create_random_tree(
                random_state, n_features, init_depth,
                method='half and half', feature_names=feature_names
            )
        else:
            r = random_state.uniform()
            parent = tournament()

            if r < method_probs[0]:
                donor = tournament()
                child = EvolutionOps.crossover(donor, parent, random_state)
            elif r < method_probs[1]:
                child = EvolutionOps.subtree_mutation(
                    parent, random_state, n_features, init_depth, feature_names
                )
            elif r < method_probs[2]: 
                child = EvolutionOps.hoist_mutation(parent, random_state)
            elif r < method_probs[3]: 
                child = EvolutionOps.point_mutation(
                    parent, random_state, p_point_replace, n_features, feature_names
                )
            else: 
                child = deepcopy(parent)

        evolution_ops_time_total += time.perf_counter() - t_evolution_start
        evaluator.calculate_fitness(child)

        if child.raw_fitness_ >= evaluator.PENALTY_VALUE:
            child.fitness_ = child.raw_fitness_
        else:
            child.fitness_ = child.raw_fitness_ + parsimony_coefficient * len(child)
        programs.append(child)

    batch_stats = {**evaluator.perfomence_stats, 'evolution_ops_time': evolution_ops_time_total}
    return programs, batch_stats


class EvolutionEngine:
    def __init__(self,
                 population_size=1000,
                 generations=20,
                 n_jobs=1,
                 tournament_size=20,
                 init_depth=(2, 6),
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 parsimony_coefficient=0.001,
                 random_state=None,
                 verbose=1):
        
        self.population_size = population_size
        self.generations = generations
        self.n_jobs = n_jobs
        self.tournament_size = tournament_size
        self.init_depth = init_depth
        self.p_point_replace = p_point_replace
        self.feature_names = None 
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.parsimony_coefficient = parsimony_coefficient

        probs = [p_crossover, p_subtree_mutation, p_hoist_mutation, p_point_mutation]
        self.method_probs = np.cumsum(probs)
        if self.method_probs[-1] > 1.0:
            raise ValueError("Total mutation probability exceeds 1.0")
        
        self._programs = []
        self.best_program = None
        self.history = []

    def fit(self, dataset_dict, model, scaler, target_coords, n_feats):
        print("Uploading data to ray obj store...")
        dataset_ref = ray.put(dataset_dict)
        model_ref = ray.put(model)
        scaler_ref = ray.put(scaler)
        target_ref = ray.put(target_coords)

        n_jobs, n_programs_per_job, _ = _partition_estimators(self.population_size, self.n_jobs)
        
        parents = None

        consecutive_failures = 0

        print(f"Starting evolution: {self.population_size} individuals, {self.generations} generations.")

        for gen in range(self.generations):
            start_time = time.time()

            seeds = self.random_state.integers(np.iinfo(np.int32).max, size=n_jobs)

            futures = []
            for i in range(n_jobs):
                futures.append(
                    _evolve_and_evaluate_batch.remote(
                        n_programs_per_job[i],
                        parents,
                        seeds[i],
                        dataset_ref,
                        model_ref,
                        scaler_ref,
                        target_ref,
                        n_feats,
                        self.init_depth,
                        self.method_probs,
                        self.tournament_size,
                        self.p_point_replace,
                        self.feature_names,
                        self.parsimony_coefficient                        
                    )
                )
            
            results_raw = ray.get(futures)
            
            population = []
            
            global_stats = {'execute_time': 0.0, 'ela_time': 0.0, 'encode_time': 0.0, 'evolution_ops_time': 0.0, 'cnt': 0}

            for batch_programs, batch_stats in results_raw:
                population.extend(batch_programs)
                
                global_stats['execute_time'] += batch_stats['execute_time']
                global_stats['ela_time']  += batch_stats['ela_time']
                global_stats['encode_time'] += batch_stats['encode_time']
                global_stats['evolution_ops_time'] += batch_stats.get('evolution_ops_time', 0.0)
                global_stats['cnt']     += batch_stats['cnt']

            if global_stats['cnt'] > 0:
                count = global_stats['cnt']
                avg_exec = (global_stats['execute_time'] / count) * 1000 
                avg_ela  = (global_stats['ela_time'] / count) * 1000   
                avg_model = (global_stats['encode_time'] / count) * 1000 
                total_evolution_s = global_stats['evolution_ops_time']
                total_ela_s = global_stats['ela_time']
                total_exec_s = global_stats['execute_time']
                total_encode_s = global_stats['encode_time']
                print(f"--- [Perf Stats] Gen {gen} ---")
                print(f"    Avg Tree Exec : {avg_exec:.4f} ms  (total {total_exec_s:.3f} s)")
                print(f"    Avg ELA Calc  : {avg_ela:.4f} ms  (total {total_ela_s:.3f} s)")
                print(f"    Avg AE Model  : {avg_model:.4f} ms  (total {total_encode_s:.3f} s)")
                print(f"    Evolution ops : total {total_evolution_s:.3f} s  (tournament+crossover/mutation/init)")
                print(f"    Total Processed: {count}  (ELA evaluations this gen: {count})")
                print(f"-----------------------------")

            valid_pop = [p for p in population if p.raw_fitness_ < 1e4]
            if not valid_pop:
                consecutive_failures += 1
                print(f'Gen {gen}: No valid individuals found')
                if not parents:
                    raise RuntimeError("Failed to gen any valid individual in Gen 0.")
                
                if consecutive_failures >= 3:
                    print("Consecutive failures detected. Restart.")
                    parents = None
                    population = None
                    consecutive_failures = 0
                    continue
                
                population = parents

            else: 
                consecutive_failures = 0
                
                current_best = min(valid_pop, key=lambda x: x.raw_fitness_)
                if self.best_program is None or current_best.raw_fitness_ < self.best_program.raw_fitness_:
                    self.best_program = deepcopy(current_best)
            avg_fitness = np.mean([p.raw_fitness_ for p in valid_pop]) if valid_pop else np.inf
            duration = time.time() - start_time
            
            time_breakdown = {}
            if global_stats['cnt'] > 0:
                time_breakdown = {
                    'evolution_ops_s': global_stats['evolution_ops_time'],
                    'execute_s': global_stats['execute_time'],
                    'ela_s': global_stats['ela_time'],
                    'encode_s': global_stats['encode_time'],
                }
            self.history.append({
                'gen': gen,
                'best_fitness': self.best_program.raw_fitness_ if self.best_program else np.inf,
                'avg_fitness': avg_fitness,
                'valid_count': len(valid_pop),
                'time': duration,
                'time_breakdown': time_breakdown,
            })                    

            if self.verbose:
                print(f"Gen {gen}: Best {self.best_program.raw_fitness_:.4f} | Avg {avg_fitness:.4f} | Valid {len(valid_pop)}/{self.population_size} | Time {duration:.2f}s")
                if self.best_program:
                    print(f"    Best Expr: {self.best_program}")
                    print(f"    Coord: {self.best_program.coordi_2D}")
                    print(f"    Target Coord: {target_coords}")
            
            parents = population

        return self.best_program, self.history