"""Stage 2 — search for periodic transit-like dips (BLS and TLS)."""
import os
import numpy as np


def bls_search(lc, min_period=0.5, max_period=12.0):
    """Box Least Squares — fast, coarse duration. Good fallback."""
    pg = lc.to_periodogram(method="bls", minimum_period=min_period,
                           maximum_period=max_period, frequency_factor=500)
    return {
        "period": float(pg.period_at_max_power.value),
        "t0": float(pg.transit_time_at_max_power.value),
        "depth": float(pg.depth_at_max_power),
        "duration": float(pg.duration_at_max_power.value),
        "power": float(pg.max_power.value),
    }


def fast_search(lc, min_period=0.5, max_period=10.0):
    """Fast two-stage search: coarse BLS on a binned curve (~0.5 s) to find the
    period, then TLS refined in a narrow ±3% window around it (~0.5-1 s).
    Returns (candidate, periodogram_data) — the BLS periodogram is reused for
    the UI chart so it is never computed twice."""
    try:
        blc = lc.bin(time_bin_size=10.0 / 1440.0)   # 10-min bins -> ~5x fewer points
    except Exception:
        blc = lc
    pg = blc.to_periodogram(method="bls", minimum_period=min_period,
                            maximum_period=max_period, frequency_factor=300)
    p0 = float(pg.period_at_max_power.value)
    lo = max(min_period, p0 * 0.97)
    hi = min(max_period * 1.2, p0 * 1.03)
    try:
        cand = tls_search(lc, min_period=lo, max_period=hi)   # narrow -> fast
    except Exception:
        cand = {   # BLS-only fallback
            "period": p0,
            "t0": float(pg.transit_time_at_max_power.value),
            "depth": float(pg.depth_at_max_power),
            "duration": float(pg.duration_at_max_power.value),
            "power": float(pg.max_power.value),
        }
    pgdata = {"p": pg.period.value, "power": pg.power.value, "best": cand["period"]}
    return cand, pgdata


def tls_search(lc, min_period=0.5, max_period=10.0):
    """Transit Least Squares — fits a realistic limb-darkened transit shape,
    giving a much sharper period & duration than BLS (key for clean depth and
    V-vs-U-shape features). Tuned for speed in bulk dataset building:
    narrower period range + lighter grids (~1.5-2x faster, tiny precision cost)."""
    from transitleastsquares import transitleastsquares
    t = np.ascontiguousarray(lc.time.value, dtype="float64")
    f = np.ascontiguousarray(lc.flux.value, dtype="float64")
    m = np.isfinite(t) & np.isfinite(f)
    res = transitleastsquares(t[m], f[m]).power(
        period_min=min_period, period_max=max_period,
        oversampling_factor=2, duration_grid_step=1.15,
        use_threads=os.cpu_count() or 2, show_progress_bar=False)
    return {
        "period": float(res.period),
        "t0": float(res.T0),
        "depth": float(abs(1.0 - res.depth)),
        "duration": float(res.duration),
        "power": float(res.SDE),
        "period_uncertainty": float(getattr(res, "period_uncertainty", float("nan"))),
    }
