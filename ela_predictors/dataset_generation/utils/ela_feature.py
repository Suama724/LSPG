"""
- problem需具 .eval(x)
- 内部对 convexity 传 problem.eval; meta/ic/dist 使用归一化后的 Ys
"""
import math

import numpy as np

from .classical_ela_feature import (
    _validate_and_convert_to_ndarray,
    calculate_ela_conv,
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_information_content,
)


def get_ela_feature(problem, Xs, Ys, random_state, ela_conv_nsample=200):
    total_calculation_time_cost = 0
    total_calculation_fes = 0
    all_features = []

    X, y = _validate_and_convert_to_ndarray(Xs, Ys)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = y_max - y_min
    if y_span <= 0:
        y_span = 1.0

    ela_conv_full_results = calculate_ela_conv(
        X,
        y,
        problem.eval,
        ela_conv_nsample=ela_conv_nsample,
        seed=random_state,
        y_min=y_min,
        y_max=y_max,
        y_span=y_span,
    )
    total_calculation_fes += ela_conv_full_results["ela_conv.additional_function_eval"]
    total_calculation_time_cost += ela_conv_full_results["ela_conv.costs_runtime"]
    for k in ela_conv_full_results.keys():
        if (k != "ela_conv.additional_function_eval") and (k != "ela_conv.costs_runtime"):
            v = ela_conv_full_results[k]
            if math.isnan(v):
                v = 0.0
            elif math.isinf(v):
                v = 1.0
            all_features.append(v)

    y_norm = (y - y_min) / y_span

    ela_meta_full_results = calculate_ela_meta(X, y_norm)
    total_calculation_time_cost += ela_meta_full_results["ela_meta.costs_runtime"]
    for k in ela_meta_full_results.keys():
        if k != "ela_meta.costs_runtime":
            v = ela_meta_full_results[k]
            if math.isnan(v):
                v = 0.0
            elif math.isinf(v):
                v = 1.0
            all_features.append(v)

    ela_ic_full_results = calculate_information_content(X, y_norm, seed=random_state)
    total_calculation_time_cost += ela_ic_full_results["ic.costs_runtime"]
    for k in ela_ic_full_results.keys():
        if k != "ic.costs_runtime":
            v = ela_ic_full_results[k]
            if v is None:
                v = 0.0
            elif math.isnan(v):
                v = 0.0
            elif math.isinf(v):
                v = 1.0
            all_features.append(v)

    ela_dis_full_results = calculate_ela_distribution(X, y_norm)
    total_calculation_time_cost += ela_dis_full_results["ela_distr.costs_runtime"]
    for k in ela_dis_full_results.keys():
        if k != "ela_distr.costs_runtime":
            v = ela_dis_full_results[k]
            if math.isnan(v):
                v = 0.0
            elif math.isinf(v):
                v = 1.0
            all_features.append(v)

    return np.array(all_features), total_calculation_fes, total_calculation_time_cost

