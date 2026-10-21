"""The PAN authorship-verification measures (NumPy re-implementation of the official evaluator).

All take ground-truth labels (1 = same author) and scores in [0, 1]; a score of exactly 0.5 is a
non-answer.

* AUC     - ROC area on the raw scores
* c@1     - accuracy that credits non-answers with the accuracy achieved on the answered problems
* F0.5u   - F0.5 that counts non-answers as false negatives
* F1      - F1 over answered problems only
* Brier   - 1 - mean squared error of the scores
* overall - mean of the five
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _arrays(true_y, pred_y):
    return np.asarray(true_y, dtype=float), np.asarray(pred_y, dtype=float)


def auc(true_y: Sequence[int], pred_y: Sequence[float]) -> float:
    y, s = _arrays(true_y, pred_y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # Mann-Whitney U with average ranks for ties
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def c_at_1(true_y: Sequence[int], pred_y: Sequence[float]) -> float:
    y, s = _arrays(true_y, pred_y)
    n = len(s)
    unanswered = s == 0.5
    correct = ((s > 0.5) == (y > 0.5)) & ~unanswered
    nc, nu = correct.sum(), unanswered.sum()
    return float((nc + nu * nc / n) / n) if n else 0.0


def f1(true_y: Sequence[int], pred_y: Sequence[float]) -> float:
    y, s = _arrays(true_y, pred_y)
    answered = s != 0.5
    y, p = y[answered], (s[answered] > 0.5).astype(float)
    tp = np.sum((p == 1) & (y == 1))
    fp = np.sum((p == 1) & (y == 0))
    fn = np.sum((p == 0) & (y == 1))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def f_05_u(true_y: Sequence[int], pred_y: Sequence[float]) -> float:
    y, s = _arrays(true_y, pred_y)
    unanswered = s == 0.5
    pred = s > 0.5
    tp = np.sum(pred & ~unanswered & (y == 1))
    fp = np.sum(pred & ~unanswered & (y == 0))
    fn = np.sum(~pred & ~unanswered & (y == 1))
    nu = unanswered.sum()
    denom = 1.25 * tp + 0.25 * (fn + nu) + fp
    return float(1.25 * tp / denom) if denom else 0.0


def brier(true_y: Sequence[int], pred_y: Sequence[float]) -> float:
    y, s = _arrays(true_y, pred_y)
    return float(1 - np.mean((s - y) ** 2)) if len(s) else 0.0


def evaluate_all(true_y: Sequence[int], pred_y: Sequence[float], digits: int = 3) -> Dict[str, float]:
    results = {"auc": auc(true_y, pred_y), "c@1": c_at_1(true_y, pred_y), "f_05_u": f_05_u(true_y, pred_y),
               "F1": f1(true_y, pred_y), "brier": brier(true_y, pred_y)}
    results["overall"] = float(np.mean(list(results.values())))
    return {k: round(v, digits) for k, v in results.items()} if digits is not None else results
