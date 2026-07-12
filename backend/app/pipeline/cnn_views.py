"""Build the CNN's global/local views from a cleaned light curve + candidate.

These MUST match how the training views were built (the Colab build notebooks):
median-binned phase-fold, then _norm so the dip bottoms out near -1. Note this
is DIFFERENT from features.extract_features (which makes coarser 201/61 views for
the RandomForest) — the CNN was trained on 2001/201 normalized views.
"""
import numpy as np


def _binned(ph, f, hw, n):
    sel = np.abs(ph) < hw
    x, y = ph[sel], f[sel]
    if len(y) < n:                                   # sparse: interpolate
        o = np.argsort(x)
        xs = x[o] if len(x) else np.array([0.0])
        ys = y[o] if len(y) else np.array([1.0])
        return np.interp(np.linspace(-hw, hw, n), xs, ys)
    bins = np.linspace(-hw, hw, n + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, n - 1)
    out = np.ones(n)
    for b in range(n):
        v = y[idx == b]
        if len(v):
            out[b] = np.median(v)
    return out


def _norm(v):
    v = v - np.median(v)
    mn = np.min(v)
    return v / abs(mn) if mn < 0 else v


def build_views(lc, cand, n_global=2001, n_local=201):
    """Return (global_view[2001], local_view[201]) as float32, matching training."""
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    period = float(cand["period"])
    t0 = float(cand["t0"])
    dur = float(cand["duration"])
    ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
    durf = dur / period if period > 0 else 0.05
    g = _norm(_binned(ph, f, 0.5, n_global))
    l = _norm(_binned(ph, f, max(2 * durf, 0.01), n_local))
    return g.astype("float32"), l.astype("float32")
