import numpy as np
import ray
import os
import datetime
from tqdm import tqdm
import pandas as pd

from problem_form.bbob_utils import gen_rotate_matrix_qr
from problem_form.bbob_problem import *
from pflacco_v1.sampling import create_initial_sample
from pflacco_v1.selected_ela_feature import get_ela_feature


def build_instance(meta_func_id,
                   dim,
                   upperbound=5.,
                   shifted=False,
                   rotated=False,
                   biased=False,
                   seed=42):
    rng = np.random.RandomState(seed)
    ub = upperbound
    lb = -1 * upperbound
    if shifted:
        shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
    else:
        shift = np.zeros(dim)
    if rotated:
        H = gen_rotate_matrix_qr(dim)
    else:
        H = np.eye(dim)
    if biased:
        bias = np.random.randint(1, 26) * 100
    else:
        bias = 0
    return eval(f'F{meta_func_id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)

@ray.remote
def process_single_task(single_task_config, Xs_ref):
    current_seed = single_task_config['seed']
    np.random.seed(current_seed)
    

    instance_id = single_task_config['instance_id']

    instance = build_instance(
    meta_func_id=single_task_config['meta_func_id'],
    dim=single_task_config['dim'],
    upperbound=single_task_config['upperbound'],
    shifted=single_task_config['do_shift'],
    rotated=single_task_config['do_rotation'],
    biased=single_task_config['do_bias'],
    seed=current_seed)
    Ys = instance.eval(Xs_ref)
    ela_feats, cost_fes, cost_time = get_ela_feature(problem=instance, Xs=Xs_ref, Ys=Ys, random_state=current_seed)
    return {
        'status': 'success',
        'instance_id': instance_id,
        'meta_func_id': single_task_config['meta_func_id'],
        'seed': single_task_config['seed'],
        'dim': single_task_config['dim'],
        'ela_feats': ela_feats,
        'cost_fes': cost_fes,
        'cost_time': cost_time
    }
    '''
    try:
        instance = build_instance(
            meta_func_id=single_task_config['meta_func_id'],
            dim=single_task_config['dim'],
            upperbound=single_task_config['upperbound'],
            shifted=single_task_config['do_shift'],
            rotated=single_task_config['do_rotation'],
            biased=single_task_config['do_bias'],
            seed=single_task_config['seed'])
        Ys = instance.eval(Xs_ref)
        ela_feats, cost_fes, cost_time = get_ela_feature(problem=instance, Xs=Xs_ref, Ys=Ys, random_state=seed)
        return {
            'status': 'success',
            'instance_id': instance_id,
            'meta_func_id': single_task_config['meta_func_id'],
            'seed': single_task_config['seed'],
            'dim': single_task_config['dim'],
            'ela_feats': ela_feats,
            'cost_fes': cost_fes,
            'cost_time': cost_time
        }

    except Exception as e:
        return {
            'status': 'failed',
            'instance_id': instance_id,
            'error': str(e)
        }
    '''
    
class DataGenerationPipeline:
    def __init__(self, cfg, seed2):
        self.output_dir_raw = cfg['output_dir']
        self.instance_num = cfg['instance_num']
        self.dim = cfg['dim']
        self.upperbound = cfg['upperbound']
        self.suit = cfg['suit']
        self.difficulty = cfg['difficulty']
        self.seed = cfg['seed'] + seed2
        self.seed_X_sample = cfg['seed_X_sample']
        self.do_rotation = cfg['do_rotation']
        self.do_shift = cfg['do_shift']
        self.do_bias = cfg['do_bias']
        self.Xs_sampling_num = cfg['X_sampling_num']
        
        if self.suit == 'bbob':
            self.meta_func_ids = cfg['bbob']
            self.difficult_func_ids = cfg['bbob_difficult']
        else:
            self.meta_func_ids = cfg['bbob_noise']
            self.difficult_func_ids = cfg['bbob_noise_difficult']

        self.time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
        self.output_dir = os.path.join(self.output_dir_raw, f"{self.dim}D_{self.time_now}")
        os.makedirs(self.output_dir, exist_ok=True) 

    def generate_configs(self):
        tasks = []

        for seed_iter in range(1, self.instance_num + 1):
            for fid in self.meta_func_ids:
                instance_seed = seed_iter + self.seed + fid * 1145
                tasks.append({
                    'instance_id': f"{self.suit}_{self.dim}D_F{fid}",
                    'meta_func_id': fid,
                    'dim': self.dim,
                    'upperbound': self.upperbound,
                    'do_shift': self.do_shift,
                    'do_rotation': self.do_rotation,
                    'do_bias': self.do_bias,
                    'seed': instance_seed,
                    'instance_index': seed_iter,
                })
        return tasks
    
    def run(self):
        if not ray.is_initialized():
            ray.init()
        print("Starting Pipeline...") 
        print(f"Dim : {self.dim}")
        print(f"Instance_num : {self.instance_num}")
        print(f"Num of CPUs : {ray.cluster_resources()['CPU']}")
        print("Creating and putting shared samples to Ray Object Store...") 
        Xs = np.array(create_initial_sample(dim=self.dim, 
                                            n = self.Xs_sampling_num, 
                                            sample_type='lhs', 
                                            lower_bound=self.upperbound*(-1), 
                                            upper_bound=self.upperbound, 
                                            seed=self.seed_X_sample))
        Xs_ref = ray.put(Xs)
        print("Shared data ready")

        tasks = self.generate_configs()   
        print(f"Total tasks : {len(tasks)}")

        futures = [process_single_task.remote(t, Xs_ref) for t in tasks]

        results = []
        errors = []

        pbar = tqdm(total=len(tasks), desc=f"Generating...")

        while futures:
            done_ids, futures = ray.wait(futures, num_returns = min(len(futures), 100), timeout=1.0)

            if done_ids:
                batch_results = ray.get(done_ids)
                for res in batch_results:
                    if res['status'] == 'success':
                        results.append(res)
                    else:
                        errors.append(res)
                pbar.update(len(done_ids))

        pbar.close()        
        self.save_final(results)
        
        if errors:
            print(f"Warning: {len(errors)} tasks failed.")
            os.makedirs(self.output_dir_raw, exist_ok=True)
            pd.DataFrame(errors).to_csv(os.path.join(self.output_dir_raw, f'errors_{self.dim}D_{self.time_now}.csv'))

        ray.shutdown()

        return pd.DataFrame(results)

    def save_final(self, results):
        df = pd.DataFrame(results)
        feats_col_name = 'ela_feats'
        if feats_col_name not in df.columns:
            raise ValueError("Bad Inputs")
        if 'status' in df.columns:
            df = df.drop(columns=['status'])
        feats_col_name = 'ela_feats'
        feats_expended = pd.DataFrame(df[feats_col_name].tolist(), index=df.index)
        feats_expended.columns = [f'ela_feat_{i}' for i in range(feats_expended.shape[1])]
        df_expended = pd.concat([df.drop(columns=[feats_col_name]), feats_expended], axis=1)

        csv_filename = os.path.join(self.output_dir, f'results.csv')
        pickle_filename = os.path.join(self.output_dir, f'results.pkl')

        df_expended.to_csv(csv_filename, index=False)
        df.to_pickle(pickle_filename)

        print(f"Saved {len(df)} records in {self.output_dir}")


if __name__ == '__main__':
    cfg = {}
    pipeline = DataGenerationPipeline(cfg)
    pipeline.run()
