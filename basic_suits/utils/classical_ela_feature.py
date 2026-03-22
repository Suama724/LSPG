import math
import numpy as np
import time

from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union, Tuple

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

IC_EPSILON_DEFAULT_NUM = 100




def _validate_and_convert_to_ndarray(
    X: Union[np.ndarray, List[List[float]]],
    y: Union[np.ndarray, List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples, n_features).")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError("y must be 1-dimensional or (n_samples, 1).")
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples.")
    return X, y

def _conv_single_eval(f: Callable, x: np.ndarray) -> float:
      x = np.asarray(x, dtype=np.float64).ravel()
      val = f(x)
      val = np.asarray(val, dtype=np.float64)
      if val.size == 0:
            return np.nan
      return float(val.ravel()[0])

def calculate_ela_conv(
      X: np.ndarray,
      y: np.ndarray,
      f: Callable[[np.ndarray], float],
      ela_conv_nsample: int = 1000,
      ela_conv_threshold: float = 1e-10,
      seed: Optional[int] = None,
      y_min: Optional[float] = None,
      y_max: Optional[float] = None,
      y_span: Optional[float] = None,
) -> Dict[str, Union[int, float]]:
      start_time = time.monotonic()
      if y_min is None:
            y_min = float(np.min(y))
      if y_max is None:
            y_max = float(np.max(y))
      if y_span is None or y_span <= 0:
            y_span = y_max - y_min
            if y_span <= 0:
                  y_span = 1.0

      if seed is not None:
            np.random.seed(seed)

      n_samples_x = X.shape[0]
      indices = np.random.randint(0, n_samples_x, size=(ela_conv_nsample, 2))
      weights = np.random.uniform(size=ela_conv_nsample)
      w = weights.reshape(-1, 1)
      X1 = X[indices[:, 0]]
      X2 = X[indices[:, 1]]
      X_new = w * X1 + (1.0 - w) * X2
      y_norm = (y - y_min) / y_span
      Y_linear = weights * y_norm[indices[:, 0]] + (1.0 - weights) * y_norm[indices[:, 1]]

      try:
            Y_new_raw = f(X_new)
            Y_new_raw = np.asarray(Y_new_raw, dtype=np.float64)
            if Y_new_raw.size == ela_conv_nsample and (Y_new_raw.ndim == 1 or (Y_new_raw.ndim == 2 and Y_new_raw.shape[1] == 1)):
                  Y_new_real = np.ravel(Y_new_raw)
            else:
                  raise TypeError("batch shape mismatch")
      except (TypeError, ValueError):
            Y_new_real = np.array([
                  _conv_single_eval(f, X_new[k])
                  for k in range(ela_conv_nsample)
            ], dtype=np.float64)
      nfev = ela_conv_nsample
      f_norm = (Y_new_real - y_min) / y_span
      delta = f_norm - Y_linear

      return {
            'ela_conv.conv_prob': np.nanmean(delta < -ela_conv_threshold),
            'ela_conv.lin_prob': np.nanmean(np.abs(delta) <= ela_conv_threshold),
            'ela_conv.lin_dev_orig': np.nanmean(delta),
            'ela_conv.lin_dev_abs': np.nanmean(np.abs(delta)),
            'ela_conv.additional_function_eval': nfev,
            'ela_conv.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds(),
      }

def calculate_ela_meta(X: np.ndarray, y: np.ndarray) -> Dict[str, Union[int, float]]:
      start_time = time.monotonic()
      n_samples, n_feat = X.shape

      # X y 线性情况
      model = linear_model.LinearRegression()
      model.fit(X, y)
      lin_simple_intercept = model.intercept_
      coef = np.abs(model.coef_)
      lin_simple_coef_min = coef.min()
      lin_simple_coef_max = coef.max()
      lin_simple_coef_max_by_min = lin_simple_coef_max / (lin_simple_coef_min + 1e-30)
      lin_simple_adj_r2 = 1 - (1 - model.score(X, y)) * (n_samples - 1) / (n_samples - n_feat - 1)

      # 包含全体交叉项 x_i*x_j（无平方项） 的情况
      poly = PolynomialFeatures(interaction_only=True, include_bias=False)
      X_interact = poly.fit_transform(X)
      model = linear_model.LinearRegression()
      model.fit(X_interact, y)
      lin_w_interact_adj_r2 = 1 - (1 - model.score(X_interact, y)) * (n_samples - 1) / (n_samples - X_interact.shape[1] - 1)

      # 无交叉项的二次情况
      X_squared = np.hstack([X, X ** 2])
      model = linear_model.LinearRegression()
      model.fit(X_squared, y)
      quad_simple_adj_r2 = 1 - (1 - model.score(X_squared, y)) * (n_samples - 1) / (n_samples - X_squared.shape[1] - 1)
      half = X_squared.shape[1] // 2
      quad_coef = np.abs(model.coef_[half:])
      quad_model_con_min = quad_coef.min()
      quad_model_con_max = quad_coef.max()
      quad_simple_cond = quad_model_con_max / (quad_model_con_min + 1e-30)

      # 含交叉项的二次情况    
      n_sq = X_squared.shape[1]
      n_interact = n_sq * (n_sq - 1) // 2
      X_quad_interact = np.empty((n_samples, n_sq + n_interact), dtype=np.float64)
      X_quad_interact[:, :n_sq] = X_squared
      col = n_sq
      for i in range(n_sq):
            n_remaining = n_sq - (i + 1)
            if n_remaining > 0:
                  X_quad_interact[:, col : col + n_remaining] = (
                        X_squared[:, i : i + 1] * X_squared[:, i + 1 :]
                  )
                  col += n_remaining
      model = linear_model.LinearRegression()
      model.fit(X_quad_interact, y)
      quad_w_interact_adj_r2 = 1 - (1 - model.score(X_quad_interact, y)) * (n_samples - 1) / (n_samples - X_quad_interact.shape[1] - 1)

      return {
            'ela_meta.lin_simple.adj_r2': lin_simple_adj_r2,
            'ela_meta.lin_simple.intercept': lin_simple_intercept,
            'ela_meta.lin_simple.coef.min': lin_simple_coef_min,
            'ela_meta.lin_simple.coef.max': lin_simple_coef_max,
            'ela_meta.lin_simple.coef.max_by_min': lin_simple_coef_max_by_min,
            'ela_meta.lin_w_interact.adj_r2': lin_w_interact_adj_r2,
            'ela_meta.quad_simple.adj_r2': quad_simple_adj_r2,
            'ela_meta.quad_simple.cond': quad_simple_cond,
            'ela_meta.quad_w_interact.adj_r2': quad_w_interact_adj_r2,
            'ela_meta.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds(),
      }


def calculate_ela_distribution(
      X: np.ndarray,
      y: np.ndarray,
      ela_distr_skewness_type: int = 3,
      ela_distr_kurtosis_type: int = 3,
) -> Dict[str, Union[int, float]]:
      """ELA Distribution features. X, y 为已校验的 ndarray。"""
      start_time = time.monotonic()
      if ela_distr_skewness_type not in range(1, 4):
            raise ValueError('Skewness type must be in [1, 3].')
      if ela_distr_kurtosis_type not in range(1, 4):
            raise ValueError('Kurtosis type must be in [1, 3].')

      y = np.asarray(y, dtype=np.float64).ravel()
      y = y[~np.isnan(y)]
      n = len(y)
      if n < 4:
            raise ValueError('At least 4 complete observations are required.')

      ym = y - np.mean(y)
      m2 = np.sum(ym ** 2)
      m3 = np.sum(ym ** 3)
      y_skewness = np.sqrt(n) * m3 / (m2 ** 1.5 + 1e-30)
      if ela_distr_skewness_type == 2:
            y_skewness = y_skewness * np.sqrt(n * (n - 1)) / (n - 2)
      elif ela_distr_skewness_type == 3:
            y_skewness = y_skewness * ((1 - 1 / n) ** 1.5)

      m4 = np.sum(ym ** 4)
      r = n * m4 / (m2 ** 2 + 1e-30)
      if ela_distr_kurtosis_type == 1:
            y_kurtosis = r - 3
      elif ela_distr_kurtosis_type == 2:
            y_kurtosis = ((n + 1) * (r - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
      else:
            y_kurtosis = r * ((1 - 1 / n) ** 2) - 3

      try:
            kernel = gaussian_kde(y)
      except Exception:
            raise
      y_std = np.std(y, ddof=1)
      low_ = np.min(y) - 3 * kernel.covariance_factor() * y_std
      upp_ = np.max(y) + 3 * kernel.covariance_factor() * y_std
      positions = np.mgrid[low_:upp_:512j]
      d = kernel(positions)
      n_d = len(d)
      index = np.arange(1, n_d - 2)
      min_index = np.array([x for x in index if d[x] < d[x - 1] and d[x] < d[x + 1]])
      min_index = np.insert(min_index, 0, 0)
      min_index = np.append(min_index, n_d)
      modemass = []
      for idx in range(len(min_index) - 1):
            a = int(min_index[idx])
            b = int(min_index[idx + 1] - 1)
            modemass.append(np.mean(d[a:b]) + abs(positions[a] - positions[b]))
      n_peaks = (np.array(modemass) > 0.1).sum()

      return {
            'ela_distr.skewness': float(y_skewness),
            'ela_distr.kurtosis': float(y_kurtosis),
            'ela_distr.number_of_peaks': int(n_peaks),
            'ela_distr.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds(),
      }

def _default_ic_epsilon() -> np.ndarray:
    return np.insert(10.0 ** np.linspace(-5, 15, IC_EPSILON_DEFAULT_NUM), 0, 0.0)

def calculate_information_content(
      X: np.ndarray,
      y: np.ndarray,
      ic_sorting: str = 'nn',
      ic_nn_neighborhood: int = 20,
      ic_nn_start: Optional[int] = None,
      ic_epsilon: Optional[Union[List[float], np.ndarray]] = None,
      ic_settling_sensitivity: float = 0.05,
      ic_info_sensitivity: float = 0.5,
      seed: Optional[int] = None,
) -> Dict[str, Union[int, float]]:

      start_time = time.monotonic()

      if ic_epsilon is None:
            ic_epsilon = _default_ic_epsilon()
      ic_epsilon = np.asarray(ic_epsilon, dtype=np.float64)
      if ic_epsilon.min() < 0:
            raise ValueError('ic_epsilon can only contain numbers in [0, inf).')
      if 0 not in ic_epsilon:
            raise ValueError("One component of ic_epsilon has to be 0.")
      if ic_sorting not in ['nn', 'random']:
            raise ValueError('ic_sorting must be "nn" or "random".')
      if ic_settling_sensitivity < 0 or ic_info_sensitivity < -1 or ic_info_sensitivity > 1:
            raise ValueError('Invalid sensitivity parameters.')
      epsilon = np.unique(ic_epsilon)

      XY = np.column_stack([X, np.ravel(y)[:X.shape[0]]])
      _, idx = np.unique(XY.view(np.dtype((np.void, XY.dtype.itemsize * XY.shape[1]))), return_index=True)
      X = X[idx]
      y = np.asarray(y, dtype=np.float64).ravel()[idx]

      X_view = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
      _, inv, counts = np.unique(X_view, return_inverse=True, return_counts=True)
      if np.all(counts[inv] > 1):
            raise ValueError('Cannot compute IC: all (X) rows are identical.')
      dup = counts[inv] > 1
      if np.any(dup):
            X_rest = X[~dup]
            y_rest = np.asarray(y, dtype=np.float64).ravel()[~dup]
            Z = X[dup]
            v = np.asarray(y, dtype=np.float64).ravel()[dup]
            Z_view = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
            _, inv_z, u_idx = np.unique(Z_view, return_inverse=True, return_index=True)
            X_agg = Z[u_idx]
            y_agg = np.array([np.mean(v[inv_z == j]) for j in range(len(u_idx))])
            X = np.vstack([X_rest, X_agg])
            y = np.concatenate([y_rest, y_agg])

      if seed is not None and isinstance(seed, int):
            np.random.seed(seed)

      n_samples = X.shape[0]
      if ic_sorting == 'random':
            permutation = np.random.permutation(n_samples)
            X = X[permutation]
            d = np.sqrt(np.sum((X[:-1] - X[1:]) ** 2, axis=1))
      else:

            if ic_nn_start is None:
                  ic_nn_start = int(np.random.randint(0, n_samples, size=1)[0])
            n_neighbors = min(ic_nn_neighborhood, n_samples)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            candidates_set = set(range(n_samples)) - {ic_nn_start}
            permutation = [ic_nn_start] + [None] * (n_samples - 1)
            dists = [None] * n_samples
            current = ic_nn_start
            for i in range(1, n_samples):
                  currents = indices[current]
                  next_c = [c for c in currents if c in candidates_set]
                  if next_c:
                        current = next_c[0]
                        permutation[i] = current
                        candidates_set.discard(current)
                        dists[i] = distances[permutation[i - 1], np.where(indices[permutation[i - 1]] == current)[0][0]]
                  else:
                        nbrs2 = NearestNeighbors(n_neighbors=1).fit(X[np.array(list(candidates_set))])
                        c_arr = np.array(list(candidates_set))
                        dist2, idx2 = nbrs2.kneighbors(X[current:current + 1])
                        current = c_arr[int(idx2[0, 0])]
                        permutation[i] = current
                        candidates_set.discard(current)
                        dists[i] = float(dist2[0, 0])
            permutation = np.array(permutation)
            d = np.array(dists[1:], dtype=np.float64)

      d = np.where(d <= 0, 1e-30, d)

      y_perm = np.asarray(y, dtype=np.float64).ravel()[permutation]
      diff_y = np.ediff1d(y_perm)
      ratio = diff_y / d
      abs_ratio = np.abs(ratio)

      psi_eps = np.where(abs_ratio[:, None] < epsilon, 0, np.sign(ratio)[:, None])
      psi_eps = psi_eps.astype(np.int8)

      a = psi_eps[:-1, :]   # 前一段状态 (n_trans-1, n_eps)
      b = psi_eps[1:, :]    # 后一段状态 (n_trans-1, n_eps)
      n_trans = psi_eps.shape[0]

      p_m1_0 = np.mean((a == -1) & (b == 0), axis=0)
      p_m1_1 = np.mean((a == -1) & (b == 1), axis=0)
      p_0_m1 = np.mean((a == 0) & (b == -1), axis=0)
      p_0_1 = np.mean((a == 0) & (b == 1), axis=0)
      p_1_m1 = np.mean((a == 1) & (b == -1), axis=0)
      p_1_0 = np.mean((a == 1) & (b == 0), axis=0)
      probs = np.stack([p_m1_0, p_m1_1, p_0_m1, p_0_1, p_1_m1, p_1_0], axis=0)  # (6, n_eps)

      tiny = 1e-300
      H = -np.sum(np.where(probs > tiny, probs * np.log(np.maximum(probs, tiny)) / np.log(6), 0.0), axis=0)

      M = np.zeros(psi_eps.shape[1], dtype=np.float64)
      if n_trans > 1:
            for j in range(psi_eps.shape[1]):
                  seq = psi_eps[:, j]
                  seq_stripped = seq[seq != 0]
                  if len(seq_stripped) > 0:
                        n_changes = np.sum(seq_stripped[:-1] != seq_stripped[1:])
                        M[j] = n_changes / (n_trans - 1)

      eps_s = epsilon[H < ic_settling_sensitivity]
      eps_s = np.log(eps_s.min()) / np.log(10) if len(eps_s) > 0 else None
      m0 = M[epsilon == 0]
      m0 = m0[0] if len(m0) > 0 else 0
      eps05_idx = np.where(M > ic_info_sensitivity * m0)[0]
      eps05 = np.log(epsilon[eps05_idx].max()) / np.log(10) if len(eps05_idx) > 0 else None

      return {
            'ic.h_max': float(np.max(H)),           # 最大信息熵
            'ic.eps_s': eps_s,
            'ic.eps_max': float(np.median(epsilon[H == np.max(H)])),  # H 达最大的 epsilon 中位数
            'ic.eps_ratio': eps05,
            'ic.m0': float(m0),
            'ic.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds(),
      }