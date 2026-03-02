import pickle
import numpy as np
import os
import ray
import os
import datetime
from net.AE import load_model

from renew_gp.engine import EvolutionEngine
from net.AE import load_model
from utils.create_initial_sample import create_initial_sample
from config import config_func_generator


class Pipeline:
    def __init__(self, cfg):

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.model_path = cfg['model_path']
        self.scaler_path = cfg['scaler_path']
        self.sample_path = cfg['sample_path']
        self.save_path = cfg['save_path']
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
        self.save_path = os.path.join(self.save_path, time_now)
        os.makedirs(self.save_path, exist_ok=True)

        self.dim  = cfg['dim']
        self.bound = cfg['bound']
        self.seed = cfg['seed']
        self.num_ela_feats = cfg['num_ela_feats']

        self.population_size = cfg['population_size']
        self.generation = cfg['generation']
        self.n_jobs = cfg['n_jobs']
        self.generate_at_indices = cfg.get('generate_at_indices') or None  # None/[] = 全部；否则只生成指定下标
        #self.n_jobs = 1

        self.model = load_model(self.model_path, ela_feats_num=self.num_ela_feats, device='cpu')

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.all_sample_points = np.load(self.sample_path).reshape(-1, 2)
        print(f"Num of all problems to generate: {len(self.all_sample_points)}.")


        print("Initializing Test Data...")
        self.dataset_dict = {}
        target_dim = self.dim
        n_samples = cfg['X_sampling_num']
        X = np.array(create_initial_sample(
            dim=target_dim, n=n_samples, sample_type='lhs',
            lower_bound=-1 * self.bound, upper_bound=self.bound,
            seed=self.seed
        ))
        self.dataset_dict[target_dim] = (X, None) # None预留了y的位置


    def get_latent_coord(self, func_id): # Start from 0
        return self.all_sample_points[func_id]
        
    def run_batch(self):
        indices = self.generate_at_indices if self.generate_at_indices else range(len(self.all_sample_points))
        for func_id in indices:
            try:
                self.solve(func_id)
            except Exception as e:
                print(f"Error when gen func {func_id}: {e}")
                continue
    
    def solve(self, func_id):
        print(f"Now generating the No.{func_id} problem.")

        log_path = os.path.join(self.save_path, "evolution_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Now generating the No.{func_id} problem.\n")
            f.flush()

        latent_coords = self.get_latent_coord(func_id)

        engine = EvolutionEngine(
            population_size=self.population_size,
            generations=self.generation,
            n_jobs=self.n_jobs,
            random_state=self.seed + func_id,
            verbose=1
        )

        best_program, history = engine.fit(
            dataset_dict=self.dataset_dict,
            model=self.model,
            scaler=self.scaler,
            target_coords=latent_coords,
            n_feats=self.dim,
            log_path=log_path,
        )
        
        self.save_single_func(func_id, best_program, history)

    def save_single_func(self, func_id, best_program, history):
        if best_program is None:
            print(f"Func {func_id} failed to produce a valid program.")
            return 
        
        prefix = f'func{func_id}_{self.dim}D'
        pickle_name = f'{prefix}_best.pickle'
        txt_name = f'{prefix}_info.txt'

        save_obj = {
            'program': best_program,
            'history': history,
            'target_coord': self.all_sample_points[func_id]
        }

        with open(os.path.join(self.save_path, pickle_name), 'wb') as f:
            pickle.dump(save_obj, f)

        with open(os.path.join(self.save_path, txt_name), 'w') as f:
            f.write(f"Function ID: {func_id}\n")
            f.write(f"Target Coord: {self.all_sample_points[func_id]}\n")
            f.write(f"Achieved Coord: {best_program.coordi_2D}\n")
            f.write(f"Fitness (Distance): {best_program.raw_fitness_}\n")
            f.write(f"Best Dimension: {best_program.best_dim}\n")
            f.write(f"Structure: {str(best_program)}\n")
            

        print(f"Saved func {func_id} to {self.save_path}")

def example_use_generated_func(program_path):
    with open(program_path, 'rb') as f:
        top_10_programs = pickle.load(f)
    best = top_10_programs[0]
    print(f"Coord {best.coordi_2D}")
    X_new = np.random.rand(5, 10)
    y_pred = best.execute(X_new)


if __name__ == '__main__':
    cfg = config_func_generator
    pipeline = Pipeline(cfg)
    pipeline.run_batch()
    #pipeline.solve(3)
