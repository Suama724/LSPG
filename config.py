import os
import torch

# 所有生成产物的根目录，读写路径均基于此
ARTIFACTS_DIR = './artifacts'

# === AE Training & Dataset Generation ===
# 选定用于训练AE的训练集
DATASET_USED_TO_TRAIN_AE = '5D_2026_02_26_104537'

# === Latent Space Sampling ===
# 选定用于采样潜空间的AE
AE_SUIT_USED_TO_SAMPLE_POINTS = '5D_2026_02_26_110633'
# 选定进行Sample的数据集
DATASET_USED_TO_SAMPLE = '5D_2026_02_26_104537'

# === Function Generation ===
# 选定进行Generate Function的sample points
SAMPLE_POINTS_USED_TO_GENERATE_FUNCTION = '2026_02_26_120027'
# 选定进行Generate Function的AE
AE_SUIT_USED_TO_GENERATE_FUNCTION = '5D_2026_02_26_110633'

config_AE = {
    'loaded_dataset': os.path.join(ARTIFACTS_DIR, 'datasets', DATASET_USED_TO_TRAIN_AE, 'results.pkl'),
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'ela_feats_num': 21,
    'epoch': 300, 
    'batch': 32,
    'save_path': os.path.join(ARTIFACTS_DIR, 'models'),
    'val_split': 0.2,
    'seed': 42,
}

config_gen_BBOB_dataset = {
    'output_dir': os.path.join(ARTIFACTS_DIR, 'datasets'),
    'instance_num': 20,
    'dim': 5,
    'upperbound': 5, # 别改
    'suit': 'bbob', # 两个选择: bbob 或 bbob_noise
    'difficulty': 'difficult',
    'do_shift': True,
    'do_rotation': True,
    'do_bias': True,
    'bbob': [i for i in range(1, 25)],
    'bbob_difficult': [3,4,6,8,9,10,11,12,14,15,19,20],
    'bbob_noise': [i for i in range(101, 131)], # the noisy mode haven't been supported yet
    'bbob_noise_difficult': [101, 105, 115, 116, 117, 119, 120, 125],
    'X_sampling_num': 250,
    'seed': 42,
    'seed_X_sample': 42062 # 别改, 确保所有评估用的X是同一套
}

config_latent_space_sampler = {
    'seed': 42,
    'n_samples': 20,
    'sampler': 'local_neighborhood_dataset', 
        # 'global_grid': sample by a grid separating the [-5, 5] space
        # 'local_neighborhood_function': sample by setting one function and offering its file
        # 'local_neighborhood_dataset': the function can be selected according to its name.
    'save_path': os.path.join(ARTIFACTS_DIR, 'latent_samples'), 
    'model_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_SAMPLE_POINTS, 'autoencoder_best.pth'), 
    'scaler_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_SAMPLE_POINTS, 'scaler.pkl'),
    # config for 'global_grid'
    'dataset_path': os.path.join(ARTIFACTS_DIR, 'datasets', DATASET_USED_TO_SAMPLE, 'results.pkl'),
    # configs for 'local_neighborhood_function' and 'local_neighborhood_dataset'
    'sample_method': 'gaussian',
        # 'gaussian': samples will be chosen according to Gaussian distribution 
        # 'grid': samples will be chosen by grid around the chosen point
    'radius': 0.5,
        # additional configs for 'local_neighborhood_function' 
    'function_path': 'Not Implement yet.',
        # additional configs for 'local_neighborhood_dataset'
    'meta_func_id': 2, # To enter which function want to be generated
}

config_func_generator = {
    'seed': 42,
    'dim': 5,
    'bound': 5,
    'num_ela_feats': 21,
    'model_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_GENERATE_FUNCTION, 'autoencoder_best.pth'),
    'scaler_path': os.path.join(ARTIFACTS_DIR, 'models', AE_SUIT_USED_TO_GENERATE_FUNCTION, 'scaler.pkl'),
    'sample_path': os.path.join(ARTIFACTS_DIR, 'latent_samples', SAMPLE_POINTS_USED_TO_GENERATE_FUNCTION, 'results.npy'), 
    'save_path': os.path.join(ARTIFACTS_DIR, 'generated_functions'),

    # configs for gp process
    'population_size': 50,
    'generation': 10,
    'n_jobs': 2
}
