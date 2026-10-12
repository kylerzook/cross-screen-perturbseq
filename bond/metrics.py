import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(y, s):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    return float(roc_auc_score(y, s))


def auprc_ratio(y, s):
    """AUPRC divided by base rate (1.0 = random)."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    return float(average_precision_score(y, s) / y.mean())


def summarize(vals):
    v = np.asarray([x for x in vals if x == x])
    return dict(mean=float(v.mean()), median=float(np.median(v)), n=int(v.size))
