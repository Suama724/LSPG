import numpy as np
import torch

from .structure import ExpressionTree
from .penalties import compute_penalty
from utils.ela_feature import get_ela_feature
from net.AE import encode_ela_feats

DEFAULT_FAILURE_PENALTY = 1e6 * 1.0


class FitnessEvaluator:
    """
    Evaluate one ExpressionTree on multiple slice-tasks.

    Each task dict supports:
    - task index in tasks list
    - X: np.ndarray, shape [n_samples, n_features]
    - target_coord: np.ndarray or None
    - target_dim: int
    """

    def __init__(
        self,
        tasks,
        model,
        scaler,
        *,
        device="cpu",
        random_state=947,
        ela_conv_nsample=230,
        w_len: float=0.0,
        w_depth: float=0.0,
        w_dim: float=0.0,
        w_symbol_cost: float=0.0,
        failure_penalty: float=DEFAULT_FAILURE_PENALTY,
    ):
        self.tasks = tasks
        self.model = model
        self.scaler = scaler
        self.device = device
        self.random_state = random_state
        self.ela_conv_nsample = ela_conv_nsample
        self.w_len = w_len
        self.w_depth = w_depth
        self.w_dim = w_dim
        self.w_symbol_cost = w_symbol_cost
        self.failure_penalty = failure_penalty

        self._perf_stats = {"cnt": 0}

    @property
    def performance_stats(self):
        return self._perf_stats

    def _distance_from_target(self, tree: ExpressionTree, task):

        '''  
        task{"X": X,
            "target_coord": target_coord,
            "target_dim": int(X.shape[1]),}
        '''
        X = np.asarray(task["X"], dtype=np.float64)
        pred = np.asarray(tree.eval(X), dtype=np.float64).ravel()
        if pred.size != X.shape[0]:
            return self.failure_penalty
        if np.isnan(pred).any() or np.isinf(pred).any() or np.std(pred) < 1e-12:
            return self.failure_penalty

        target_coord = task.get("target_coord", None)
        if target_coord is None:
            return self.failure_penalty

        ela_feats, _, _ = get_ela_feature(
            tree,
            X,
            pred,
            random_state=self.random_state,
            ela_conv_nsample=self.ela_conv_nsample,
        )
        ela_feats = np.asarray(ela_feats, dtype=np.float64).ravel()
        if np.isnan(ela_feats).any() or np.isinf(ela_feats).any():
            return self.failure_penalty

        ela_scaled = self.scaler.transform(ela_feats.reshape(1, -1))

        latent = encode_ela_feats(
            self.model,
            torch.tensor(ela_scaled, dtype=torch.float32),
            device=self.device,
        )
        latent = np.asarray(latent, dtype=np.float64).ravel()
        target_coord = np.asarray(target_coord, dtype=np.float64).ravel()
        if latent.shape != target_coord.shape:
            return self.failure_penalty
        if np.isnan(latent).any() or np.isinf(latent).any():
            return self.failure_penalty
        return float(np.sqrt(np.mean((latent - target_coord) ** 2)))

    def calculate_fitness(self, tree: ExpressionTree):
        self._perf_stats["cnt"] += 1
        raw_per_task = []  # total fitness per task
        raw_dist_list = []  # raw distance per task
        penalty_list = []  # penalty per task

        for task in self.tasks:
            target_dim = int(task.get("target_dim", 0))
            try:
                raw_dist = self._distance_from_target(tree, task)
                penalty = compute_penalty(
                    tree,
                    w_len=self.w_len,
                    w_depth=self.w_depth,
                    w_dim=self.w_dim,
                    w_symbol_cost=self.w_symbol_cost,
                    target_dim=target_dim,
                )
                total = float(raw_dist + penalty)
            except Exception:
                raw_dist = self.failure_penalty
                penalty = self.failure_penalty
                total = self.failure_penalty

            raw_dist_list.append(float(raw_dist))
            penalty_list.append(float(penalty))
            raw_per_task.append(float(total))

        best_idx = int(np.argmin(raw_per_task)) if raw_per_task else 0
        # task index itself is the task id/order
        best_task_id = int(best_idx)
        best_raw_dist = float(raw_dist_list[best_idx]) if raw_dist_list else self.failure_penalty
        scaler_fitness = float(raw_per_task[best_idx]) if raw_per_task else self.failure_penalty

        tree.raw_per_task = raw_per_task
        tree.raw_dist_per_task = raw_dist_list
        tree.penalty_per_task = penalty_list
        tree.best_task_id = best_task_id
        tree.skill_factor = best_task_id
        tree.raw_dist = best_raw_dist
        tree.scaler_fitness = scaler_fitness
        tree.fitness = scaler_fitness
        return scaler_fitness