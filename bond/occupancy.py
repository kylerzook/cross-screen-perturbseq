"""Statistical-mechanics occupancy.

Single site:   P(bound) = sigmoid(mu - E + log a)
  E  : relative binding energy (kT) from sequence
  mu : TF chemical potential (kT) ~ log active-TF concentration; slides all sites together
  a  : accessibility in (0, 1]; nucleosome competition lowers effective affinity (soft, not a mask)
Promoter occupancy = expected number of bound TF molecules = sum over positions and strands.
"""
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def occupancy(E, mu=0.0, log_acc=None):
    """E: (G, 2, P). log_acc: None or (G, P) aligned to motif start positions."""
    x = mu - E
    if log_acc is not None:
        x = x + log_acc[:, None, :]
    return sigmoid(x).sum(axis=(1, 2))
