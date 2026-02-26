import datetime
import pickle
import os
import numpy as np
import pandas as pd
import torch

from config import config_latent_space_sampler
import net.AE as AE
from vis.latent_plots import plot_sample_latent_space, plot_dataset_latent_space
    
class GlobalGridSampler:
    def sample(self, n_samples, dataset_encoded_points):
        min_x, min_y = np.min(dataset_encoded_points, axis=0)
        max_x, max_y = np.max(dataset_encoded_points, axis=0)

        aspect_ratio = (max_x - min_x) / (max_y - min_y)
        n_cols = int(np.sqrt(n_samples * aspect_ratio))
        n_rows = int(n_samples / n_cols)

        x = np.linspace(min_x, max_x, n_cols)
        y = np.linspace(min_y, max_y, n_rows)
        grid_x, grid_y = np.meshgrid(x, y)
        candidates = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        return candidates

class LocalNeighborhoodSampler:
    def sample(self, center_point, n_samples, radius=0.5, method='gaussian'):
        center_point = np.array(center_point).reshape(1, -1)
        
        if method == 'grid':
            x = np.linspace(center_point[0,0]-radius, center_point[0,0]+radius, int(np.sqrt(n_samples)))
            y = np.linspace(center_point[0,1]-radius, center_point[0,1]+radius, int(np.sqrt(n_samples)))
            gx, gy = np.meshgrid(x, y)
            return np.vstack([gx.ravel(), gy.ravel()]).T
            
        elif method == 'gaussian':
            return center_point + np.random.normal(0, radius, (n_samples, 2))

class SamplePipeline:

    # Used to encode a func/program obj into the latent_space
    def encode(self):
        # 前面的区域, 请以后再来探索吧
        raise NotImplementedError

    def __init__(self, config_latent_space_sample):
        cfg = config_latent_space_sampler
        self.sampler_chosen = cfg['sampler']

        self.save_path = cfg['save_path']
        self.model_path = cfg['model_path']
        self.scaler_path = cfg['scaler_path']
        self.dataset_path = cfg['dataset_path']
        self.function_path = cfg['function_path'] # Haven't been implemented yet


        if  self.sampler_chosen == 'global_grid':
            self.sampler = GlobalGridSampler()
            self.dataset_encoded_points, _ = SamplePipeline.encode_dataset(self.dataset_path, 
                                                                        self.model_path,
                                                                        self.scaler_path)
        elif self.sampler_chosen == 'local_neighborhood_dataset':
            self.sampler = LocalNeighborhoodSampler()
            self.dataset_encoded_points, self.raw_df = self.encode_dataset(self.dataset_path, 
                                                                self.model_path,
                                                                self.scaler_path,
                                                                return_df=True) 
        else:
            raise KeyError("Sampler chosen doesn't exist.")


        self.seed = cfg['seed']
        self.n_samples = cfg['n_samples'] 
        self.sample_method = cfg['sample_method']
        self.radius = cfg['radius']
        self.meta_func_id = cfg['meta_func_id']
        self.gen_sample_plot = cfg.get('gen_sample_plot', True)
         
    @staticmethod
    def encode_dataset(dataset_path, model_path, scaler_path, return_df=False):
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_pickle(dataset_path)
        feats = np.stack(df['ela_feats'].values)
        n_samples, n_feats = feats.shape
        print(f"Dataset loaded shape: {feats.shape}")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)    
        feats_normalized = scaler.transform(feats)

        device = 'cpu'
        model = AE.load_model(model_path, n_feats, device=device)
        print("Encoding...")

        encoded_points = AE.encode_ela_feats(model, feats_normalized, device=device)
        
        if return_df:
            return encoded_points, df
        return encoded_points, None

    def get_center_point_for_func(self, func_id):
        mask = self.raw_df['meta_func_id'] == func_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f"No instances found for function ID {func_id}")
            
        target_encoded = self.dataset_encoded_points[indices]
        center = np.mean(target_encoded, axis=0)
        print(f"Center point for F{func_id} calculated from {len(indices)} instances: {center}")
        return center
        



    
    # Used to save sampled points
    @staticmethod 
    def save_result(sample_results, save_path,
                    dataset_path,
                    n_samples,
                    model_path,
                    scaler_path,
                    sampler_chosen
                    ):
        os.makedirs(save_path, exist_ok=True)

        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
        save_path = os.path.join(save_path, time_now)
        os.makedirs(save_path, exist_ok=True)
        file_name = f'results.npy'
        full_path = os.path.join(save_path, file_name)
    
        np.save(full_path, sample_results)

        dataset_id = dataset_path.split('/')[-2]
        model_id = model_path.split('/')[-2]
        log = f'''
'dataset': {dataset_id},
'n_samples': {n_samples},
'model': {model_id},
'sampler_chosen': {sampler_chosen},
        '''
        with open(os.path.join(save_path, 'log.txt'), 'w') as f:
            f.write(log)   
        
        plot_dataset_latent_space(dataset_path, model_path, scaler_path, save_dir=save_path)
        plot_sample_latent_space(data=sample_results, save_dir=save_path)

        print(f"Sampled points saved to {full_path}. Shape: {sample_results.shape}")




    def run(self):
        print(f"Begin Sampling, method: {self.sampler_chosen}.")

        sample_results = None

        if self.sampler_chosen == 'global_grid':
            sample_results = self.sampler.sample(n_samples=self.n_samples, 
                                                 dataset_encoded_points=self.dataset_encoded_points)
            
        elif self.sampler_chosen == 'local_neighborhood_dataset':
            center_point = self.get_center_point_for_func(self.meta_func_id)
            sample_results = self.sampler.sample(center_point=center_point, 
                                                 n_samples=self.n_samples, 
                                                 radius=self.radius, 
                                                 method=self.sample_method)
            
        self.save_result(sample_results, self.save_path,
                        self.dataset_path,
                        self.n_samples,
                        self.model_path,
                        self.scaler_path,
                        self.sampler_chosen,)
        print(f"Finished. Saved in {self.save_path}")

if __name__ == '__main__':
    cfg = config_latent_space_sampler
    pipeline = SamplePipeline(cfg)
    pipeline.run()
