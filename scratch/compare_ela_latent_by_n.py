"""
对指定 pickle 中的 ExpressionTree, 用不同 X 样本数 N 算 ELA 并编码，比较潜空间坐标。
"""
import os
import sys
import pickle
import numpy as np
import torch


PICKLE_PATH = "artifacts/generated_functions/2026_03_01_111158/func0_10D_best.pickle"
N_LIST = [250, 500, 1000, 2500, 5000]  # 要比较的 X 样本数，用同一套 X 的前 N 行
DIM = 10
BOUND = 5
SEED = 42
MODEL_PATH = "artifacts/models/5D_2026_02_26_110633/autoencoder_best.pth"
SCALER_PATH = "artifacts/models/5D_2026_02_26_110633/scaler.pkl"
NUM_ELA_FEATS = 21


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.create_initial_sample import create_initial_sample
from utils.ela_feature import get_ela_feature
from net.AE import load_model, encode_ela_feats


class TreeAsProblem:
    def __init__(self, program):
        self.program = program

    def eval(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.asarray(self.program.execute(x), dtype=np.float64).ravel()


def main():
    pickle_path = os.path.join(PROJECT_ROOT, PICKLE_PATH) if not os.path.isabs(PICKLE_PATH) else PICKLE_PATH
    model_path = os.path.join(PROJECT_ROOT, MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH
    scaler_path = os.path.join(PROJECT_ROOT, SCALER_PATH) if not os.path.isabs(SCALER_PATH) else SCALER_PATH

    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)
    program = obj['program']
    target_coord = np.asarray(obj['target_coord']).ravel()

    model = load_model(model_path, ela_feats_num=NUM_ELA_FEATS, device='cpu')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    n_list = sorted(N_LIST)
    n_max = max(n_list)
    X_full = np.array(create_initial_sample(
        dim=DIM, n=n_max, sample_type='lhs',
        lower_bound=-BOUND, upper_bound=BOUND,
        seed=SEED,
    ), dtype=np.float64)
    problem = TreeAsProblem(program)

    results = []
    for n in n_list:
        X = X_full[:n]
        Y = problem.eval(X)
        if np.isnan(Y).any() or np.isinf(Y).any() or np.std(Y) < 1e-10:
            results.append((n, None, None))
            continue
        ela_feats, _, _ = get_ela_feature(problem, X, Y, random_state=SEED)
        ela_scaled = scaler.transform(ela_feats.reshape(1, -1))
        latent = encode_ela_feats(model, torch.tensor(ela_scaled, dtype=torch.float32), device='cpu')
        latent = np.asarray(latent).ravel()
        dist = float(np.linalg.norm(latent - target_coord))
        results.append((n, latent, dist))

    print("Target:", target_coord)
    print()
    print(f"{'N':>8} | {'latent':^28} | {'dist_to_target':>14}")
    print("-" * 54)
    for n, latent, dist in results:
        if latent is None:
            print(f"{n:>8} | (skip)")
            continue
        lat_str = np.array2string(latent, precision=4, separator=', ')
        print(f"{n:>8} | {lat_str:^28} | {dist:>14.6f}")


if __name__ == '__main__':
    main()
