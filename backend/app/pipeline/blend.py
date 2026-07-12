"""Stage 4 — centroid / blend check, computed straight from the light curve's own
centroid columns (mom_centr) — NO pixel-file download needed. ~1 ms per star.

Idea: during a real planet transit the star's light-centre stays put; for a blend
(a nearby eclipsing binary leaking in) the light-centre shifts toward that source.
Returns the in-transit vs out-of-transit centroid shift as a SIGNIFICANCE (sigma):
  ~0 = on-target (planet-like) · large = off-target (blend).
"""
import numpy as np


def _col(lc, *names):
    for n in names:
        if n in lc.columns:
            return np.asarray(lc[n].value, float)
    return None


def centroid_shift(raw_lc, candidate):
    col = _col(raw_lc, "centroid_col", "mom_centr1", "pos_corr1")
    row = _col(raw_lc, "centroid_row", "mom_centr2", "pos_corr2")
    if col is None or row is None:
        return float("nan")
    t = np.asarray(raw_lc.time.value, float)
    p = float(candidate["period"]); t0 = float(candidate["t0"])
    durf = float(candidate["duration"]) / p if p > 0 else 0.05
    ph = ((t - t0 + 0.5 * p) % p) / p - 0.5
    intr = np.abs(ph) < 0.5 * durf
    oot = np.abs(ph) > 1.5 * durf
    g = np.isfinite(col) & np.isfinite(row)
    if (intr & g).sum() < 3 or (oot & g).sum() < 3:
        return float("nan")
    scatter = np.hypot(np.std(col[oot & g]), np.std(row[oot & g])) + 1e-9
    shift = np.hypot(np.median(col[intr & g]) - np.median(col[oot & g]),
                     np.median(row[intr & g]) - np.median(row[oot & g]))
    return float(shift / scatter)   # significance (sigma)
