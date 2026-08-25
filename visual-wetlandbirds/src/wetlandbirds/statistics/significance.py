from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, exact_threshold: int = 25) -> dict:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("McNemar inputs must have equal lengths")
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    discordant = b01 + b10
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
        test = "no_discordant_pairs"
    elif discordant < exact_threshold:
        p_value = float(stats.binomtest(min(b01, b10), discordant, 0.5).pvalue)
        statistic = float(min(b01, b10))
        test = "exact_binomial"
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, 1))
        test = "chi2_continuity_corrected"
    return {"b01": b01, "b10": b10, "n_discordant": discordant, "statistic": statistic, "p_value": p_value, "test": test}


def holm_bonferroni(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted
