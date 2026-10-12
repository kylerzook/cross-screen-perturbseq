"""Fit TF chemical potentials (mu) and regulatory weights (w) from expression.

    y[g, c] = intercept + sum_t w_t * occ[g, t, c]
    occ[g, t, c] = sum_{sites k of (g,t)} sigmoid(mu[t, c] + L[k, c]),   L = -E + log(accessibility)

Sites are stored sparsely (only E below a cutoff; the rest contribute ~0).
mu_mode: 'state' -> one mu per (TF, condition); 'shared' -> one mu per TF; 'fixed' -> mu = 0.
intercept: 'gene' -> b_g;  'state' -> b_c.
Solved with L-BFGS-B, analytic gradients, L2 penalties, mu bounded.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from .occupancy import sigmoid


class OccupancyModel:
    def __init__(self, eg, et, L, G, T, mu_mode="state", intercept="gene",
                 lam_w=1e-3, lam_mu=1e-3, mu_bound=6.0, mu0=0.0, mu_center=0.0):
        self.eg, self.et, self.L = eg, et, L.astype(np.float32)
        self.n, self.C = L.shape
        self.G, self.T = G, T
        self.mu_mode, self.intercept = mu_mode, intercept
        self.lam_w, self.lam_mu, self.mu_bound = lam_w, lam_mu, mu_bound
        self.mu0 = np.broadcast_to(np.asarray(mu0, float), (T,)).copy()
        self.mu_center = np.broadcast_to(np.asarray(mu_center, float), (T,)).copy()
        ones = np.ones(self.n)
        self.A = sp.csr_matrix((ones, (eg, np.arange(self.n))), shape=(G, self.n))
        self.B = sp.csr_matrix((ones, (et, np.arange(self.n))), shape=(T, self.n))
        self.nb = G if intercept == "gene" else self.C
        self.nmu = {"state": T * self.C, "shared": T, "fixed": 0}[mu_mode]

    def unpack(self, p):
        b = p[:self.nb]; w = p[self.nb:self.nb + self.T]
        m = p[self.nb + self.T:]
        if self.mu_mode == "state":
            mu = m.reshape(self.T, self.C)
        elif self.mu_mode == "shared":
            mu = np.repeat(m[:, None], self.C, axis=1)
        else:
            mu = np.repeat(self.mu_center[:, None], self.C, axis=1)
        return b, w, mu

    def occ(self, mu):
        S = sigmoid(mu[self.et] + self.L)  # n x C
        return S

    def predict(self, p, rows=None):
        b, w, mu = self.unpack(p)
        S = self.occ(mu)
        pred = self.A @ (w[self.et][:, None] * S)
        pred = pred + (b[:, None] if self.intercept == "gene" else b[None, :])
        return pred

    def pair_occupancy(self, p):
        """Dense (G, T, C) occupancy under fitted mu."""
        _, _, mu = self.unpack(p)
        S = self.occ(mu)
        pair = self.eg * self.T + self.et
        M = sp.csr_matrix((np.ones(self.n), (pair, np.arange(self.n))), shape=(self.G * self.T, self.n))
        return np.asarray(M @ S).reshape(self.G, self.T, self.C)

    def loss_grad(self, p, y, mask):
        b, w, mu = self.unpack(p)
        S = self.occ(mu)
        pred = self.A @ (w[self.et][:, None] * S)
        pred = pred + (b[:, None] if self.intercept == "gene" else b[None, :])
        r = (pred - y) * mask
        N = mask.sum()
        loss = 0.5 * (r ** 2).sum() / N + self.lam_w * (w ** 2).sum() + self.lam_mu * ((mu - self.mu_center[:, None]) ** 2).sum() / max(self.C, 1)
        R = r[self.eg]  # n x C
        gb = (r.sum(1) if self.intercept == "gene" else r.sum(0)) / N
        gw = np.asarray(self.B @ (R * S).sum(1)).ravel() / N + 2 * self.lam_w * w
        gmu_full = w[:, None] * np.asarray(self.B @ (R * S * (1 - S))) / N + 2 * self.lam_mu * (mu - self.mu_center[:, None]) / max(self.C, 1)
        if self.mu_mode == "state":
            gm = gmu_full.ravel()
        elif self.mu_mode == "shared":
            gm = gmu_full.sum(1)
        else:
            gm = np.zeros(0)
        return loss, np.concatenate([gb, gw, gm])

    def fit(self, y, mask=None, w0=None, seed=0, maxiter=500, p_init=None):
        rng = np.random.default_rng(seed)
        if mask is None:
            mask = np.ones_like(y)
        b0 = (np.nanmean(y, 1) if self.intercept == "gene" else np.nanmean(y, 0))
        w0 = rng.normal(0, 0.1, self.T) if w0 is None else w0
        m0 = {'state': np.repeat(self.mu0, self.C), 'shared': self.mu0, 'fixed': np.zeros(0)}[self.mu_mode]
        p0 = np.concatenate([b0, w0, m0])
        mc = {'state': np.repeat(self.mu_center, self.C), 'shared': self.mu_center, 'fixed': np.zeros(0)}[self.mu_mode]
        if p_init is not None:
            p0 = p_init.copy()
        bounds = [(None, None)] * (self.nb + self.T) + [(c - self.mu_bound, c + self.mu_bound) for c in mc]
        yy = np.nan_to_num(y)
        res = minimize(self.loss_grad, p0, args=(yy, mask), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=maxiter))
        self.p, self.res = res.x, res
        return self


def sparse_sites(E, cutoff):
    """E: (G, 2, P) for one TF -> (gene_idx, flat_pos_idx, E) for sites with E < cutoff."""
    g, s, pos = np.nonzero(E < cutoff)
    return g, s * E.shape[2] + pos, E[g, s, pos]
