import numpy as np
import torch
import math
from copy import deepcopy 

from utils.ela_feature import get_ela_feature
from .structure import ExpressionTree
from net.AE import encode_ela_feats, AutoEncoder

import time

class FunctionWrapper:
    def __init__(self, tree_func):
        self.func = tree_func
    def eval(self, x):
        return self.func(x)
    
class FitnessEvaluator:
    def __init__(self, 
                 dataset_dict,
                 model: AutoEncoder,
                 scaler,
                 target_coords,
                 device='cpu'):
        self.dataset_dict = dataset_dict
        self.model = model
        self.scaler = scaler
        self.target_coords = target_coords
        self.device = device

        self.PENALTY_VALUE = 1e5

        if self.model:
            self.model.eval()
            self.model.to(device)

        self.perfomence_stats = {
            'execute_time': 0.0,
            'ela_time': 0.0,
            'encode_time': 0.0,
            'cnt': 0
        }

    def validate(self, y_pred):
        """
        合法性检查 (Sanity Check)。
        如果不合法，直接返回 False，跳过昂贵的 ELA 计算。
        """
        # 检查 NaN / Inf
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return False
        
        # 检查极大溢出
        if np.max(np.abs(y_pred)) > 1e10:
            return False

        # 检查是否为常数函数
        if np.std(y_pred) < 1e-6:
            return False
            
        return True
    
    def calculate_fitness(self, tree: ExpressionTree):
        scores = []
        meta_info = []

        local_exec_time = 0.0
        local_ela_time = 0.0
        local_model_time = 0.0
        full_success = False 

        for dim, (X_input, _) in self.dataset_dict.items():
            t_start = time.perf_counter()
            try:
                y_pred = tree.execute(X_input)
                local_exec_time += time.perf_counter() - t_start
            except Exception:
                local_exec_time += time.perf_counter() - t_start
                scores.append(self.PENALTY_VALUE)
                meta_info.append(None)
                continue

            if not self.validate(y_pred):
                scores.append(self.PENALTY_VALUE)
                meta_info.append(None)
                continue

            wrapper = FunctionWrapper(tree.execute)

            t_start = time.perf_counter()
            try:
                ela_feats, _, _ = get_ela_feature(
                    problem=wrapper,
                    Xs=X_input,
                    Ys=y_pred,
                    random_state=42,
                    ela_conv_nsample=50,
                )

                local_ela_time += time.perf_counter() - t_start

                if np.isnan(ela_feats).any() or np.isinf(ela_feats).any():
                    scores.append(self.PENALTY_VALUE)
                    meta_info.append(None)
                    continue
            except Exception:
                local_ela_time += time.perf_counter() - t_start
                scores.append(self.PENALTY_VALUE)
                meta_info.append(None)
                continue

            t_start = time.perf_counter()
            try:
                ela_feats_reshaped = ela_feats.reshape(1, -1)
                ela_scaled = self.scaler.transform(ela_feats_reshaped)

                input_tensor = torch.tensor(ela_scaled, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    encoded_coord = encode_ela_feats(model=self.model,
                                                     input_features=input_tensor,
                                                     device=self.device)
                
                if np.isnan(encoded_coord).any():
                    scores.append(self.PENALTY_VALUE)
                    meta_info.append(None)
                    continue
                local_model_time += time.perf_counter() - t_start
                full_success = True

            except Exception:
                local_model_time += time.perf_counter() - t_start
                scores.append(self.PENALTY_VALUE)
                meta_info.append(None)
                continue

            dist = np.linalg.norm(encoded_coord - self.target_coords)
            
            scores.append(dist)
            meta_info.append((dim, ela_feats, encoded_coord))

        self.perfomence_stats['execute_time'] += local_exec_time
        self.perfomence_stats['ela_time'] += local_ela_time
        self.perfomence_stats['encode_time'] += local_model_time
        self.perfomence_stats['cnt'] += 1

        if all(s == self.PENALTY_VALUE for s in scores):
            tree.raw_fitness_ = self.PENALTY_VALUE
            return self.PENALTY_VALUE
        
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_info = meta_info[best_idx]
        
        tree.raw_fitness_ = best_score
        
        if best_info is not None:
            tree.best_dim = best_info[0]
            tree.ela_feature = best_info[1]
            tree.coordi_2D = best_info[2]
        
        return best_score