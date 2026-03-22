from pathlib import Path

import numpy as np
import torch

NET_LIST = ("MLP", "SetTransformer")
CURRENT_NET = "MLP"
NUM_EVALUATE = 230
RANDOM_FUNC_NUM = 50
X_PATH = "./dataset_generation/dataset/ela_predictor_100d/X.npy"
MLP_CKPT_PATH = "./surrogate/output/mlp_surrogate_best.pth"
TF_CKPT_PATH = "./surrogate/output/set_transformer_surrogate_best.pth"


from surrogate_tester import compute_and_encode_ela, sample_random_funcs
from surrogate_tester import predict_ela_with_net
from surrogate_tester import compare_coord

if __name__ == "__main__":
    if CURRENT_NET not in NET_LIST:
        raise ValueError("Net not defined")

    x_arr = np.asarray(np.load(Path(X_PATH)))


    num_evaluate = int(x_arr.shape[0])
    problem_dim = int(x_arr.shape[1])
    if num_evaluate != NUM_EVALUATE:
        raise ValueError(
            f"NUM_EVALUATE={NUM_EVALUATE} does not match X rows={num_evaluate}"
        )

    if CURRENT_NET == "MLP":
        from surrogate import MLPSurrogate

        net = MLPSurrogate(NUM_EVALUATE)
        state_dict = torch.load(MLP_CKPT_PATH, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        desc = "MLP"

    elif CURRENT_NET == "SetTransformer":
        from surrogate import SetTransformerSurrogate

        net = SetTransformerSurrogate()
        state_dict = torch.load(TF_CKPT_PATH, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        desc = "SetTransformer"

    func_list = sample_random_funcs(RANDOM_FUNC_NUM, problem_dim)

    coords_pred = predict_ela_with_net(func_list=func_list, net=net, x_path=X_PATH)

    coords_compute = compute_and_encode_ela(func_list=func_list, x_path=X_PATH)

    compare_coord(coords_compute, coords_pred, problem_dim, desc)