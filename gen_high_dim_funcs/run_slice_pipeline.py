import numpy as np
import os

#from config import config_slice_ela_gen
from func_slice_sample_gen_pipeline import SliceELAGenPipeline

from test_high_dim_func_factory import get_problem

'''
def get_problem():
    """
    返回 (problem, full_dim)
    problem 需实现 .eval(x), x 为 (n, full_dim) 的数组 
    """
'''


ARTIFACTS_DIR = './outputs'
AE_SUIT_USED_TO_GENERATE_FUNCTION = '100D_2026_03_06_132400'
config_slice_ela_gen = {
    'save_path': os.path.join(ARTIFACTS_DIR, 'sliced_functions_from_LSPG'),
    'model_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_GENERATE_FUNCTION, 'autoencoder_best.pth'),
    'scaler_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_GENERATE_FUNCTION, 'scaler.pkl'),
    'num_ela_feats': 21,
    'bound': 5.0,
    'seed': 42,
    'X_sampling_num_ela': 230,
    'population_size': 60,
    'generation': 6,
    'n_jobs': 30,
    'slice_len': 100,
    'fill_value': 0.0,
}



def main():
    cfg = config_slice_ela_gen
    problem, full_dim = get_problem(1)

    pipeline = SliceELAGenPipeline(
        problem=problem,
        full_dim=full_dim,
        **cfg
    )
    summed = pipeline.run()
    print(f"Pipeline finished. Output directory: {pipeline.save_path}")
    return summed


if __name__ == "__main__":
    main()
