"""Metricas comparaveis entre profundidade e ocupacao estimadas e ideais."""

import numpy as np


def depth_metrics(estimated_depth_m, reference_depth_m, min_depth_m=0.1):
    """Calcula MAE, RMSE e AbsRel nos pixels validos das duas fontes."""

    estimated = np.asarray(estimated_depth_m, dtype=np.float64)
    reference = np.asarray(reference_depth_m, dtype=np.float64)
    if estimated.shape != reference.shape:
        raise ValueError("as profundidades devem possuir a mesma forma")
    valid = (
        np.isfinite(estimated)
        & np.isfinite(reference)
        & (estimated >= min_depth_m)
        & (reference >= min_depth_m)
    )
    if not np.any(valid):
        return {"valid_pixels": 0, "mae_m": np.nan, "rmse_m": np.nan, "abs_rel": np.nan}
    error = estimated[valid] - reference[valid]
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "abs_rel": float(np.mean(np.abs(error) / reference[valid])),
    }


def occupancy_metrics(estimated_grid, reference_grid):
    """Compara voxels ocupados e destaca obstaculos estimados como livres."""

    estimated_occupied = estimated_grid.occupied_voxels()
    reference_occupied = reference_grid.occupied_voxels()
    estimated_free = estimated_grid.free_voxels()

    true_positive = len(estimated_occupied & reference_occupied)
    false_positive = len(estimated_occupied - reference_occupied)
    false_negative = len(reference_occupied - estimated_occupied)
    false_free = len(reference_occupied & estimated_free)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    union = len(estimated_occupied | reference_occupied)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / precision_denominator if precision_denominator else np.nan,
        "recall": true_positive / recall_denominator if recall_denominator else np.nan,
        "iou": true_positive / union if union else np.nan,
        "false_free_rate": false_free / len(reference_occupied) if reference_occupied else np.nan,
    }
