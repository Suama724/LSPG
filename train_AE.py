from net import dataset_generate
import config
import pickle
from net.AE import split_data, normalize_data, make_dataset, AutoEncoder, train_autoencoder
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

def get_dataset(seed2=0):
    # seed2 is designed to make the random dataset_generation process more convenience  
    config_ = config.config_gen_BBOB_dataset
    #for iter_dim in [40, 50, 100, 200, 500, 1000]:
        #config_iter['dim'] = iter_dim
    pipline = dataset_generate.DataGenerationPipeline(cfg=config_, seed2=24)
    pipline.run()

def train_AE():
    cfg = config.config_AE
    dataset_path = cfg['loaded_dataset']
    data_info = pd.read_pickle(dataset_path)
    current_dim = data_info['dim'].iloc[0]

    time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    batch = cfg['batch']
    epochs = cfg['epoch']
    save_path = os.path.join(cfg['save_path'], f'{current_dim}D_{time_now}')
    os.makedirs(save_path, exist_ok=True)

    val_split = cfg['val_split']
    ela_feats_num = cfg['ela_feats_num']
    device = cfg['device']
    seed = cfg['seed']
    
    feats_set = pd.DataFrame(data_info['ela_feats'].to_list()).fillna(0)
    feats_set_array = feats_set.values.astype(np.float32)
    print(f"Shape: {feats_set_array.shape}")
    
    train_data, val_data = split_data(feats_set_array, val_split, seed)
    normalized_train_data, scaler = normalize_data(train_data)
    normalized_test_data = scaler.transform(val_data)
    
    with open(os.path.join(save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f, 0)
    
    train_loader, val_loader = make_dataset(normalized_train_data, normalized_test_data)
    model = AutoEncoder(ela_feats_num)
    train_autoencoder(model, train_loader, val_loader, 
                      save_path, save_path, device, epochs)
    
    
if __name__ == '__main__':
    #get_dataset()
    train_AE()
