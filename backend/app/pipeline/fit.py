"""Fast light-curve fitting for parameter UNCERTAINTIES (no MCMC, no heat).

Fits a trapezoid transit model to the phase-folded curve with scipy.curve_fit and
reads 1-sigma errors straight off the covariance matrix — milliseconds per star.
Gives depth ± and duration ± directly; period ± comes from TLS (period_uncertainty).
For rigorous Bayesian posteriors use the batman+emcee Colab notebook instead."""
import numpy as np
from scipy.optimize import curve_fit


def _trapezoid(ph, t0, depth, half, ingress, base):
    """Symmetric trapezoid: flat core of depth, linear ingress/egress, flat baseline.
    half = half of total duration (phase units); ingress = ingress width (phase)."""
    x = np.abs(ph - t0)
    flat = max(half - ingress, 1e-6)
    f = np.full_like(ph, base)
    core = x <= flat
    wing = (x > flat) & (x < half)
    f[core] = base - depth
    f[wing] = base - depth * (1.0 - (x[wing] - flat) / (half - flat + 1e-12))
    return f


def fit_transit(lc, cand):
    """Return {period, duration, depth} each as {value, err, unit}, or None."""
    period = float(cand["period"])
    if period <= 0:
        return None
    t0c = float(cand["t0"])
    durf = float(cand["duration"]) / period
    t = np.asarray(lc.time.value, float)
    fl = np.asarray(lc.flux.value, float)
    good = np.isfinite(t) & np.isfinite(fl)
    t, fl = t[good], fl[good]
    ph = ((t - t0c + 0.5 * period) % period) / period - 0.5

    win = min(0.5, max(4.0 * durf, 0.02))
    sel = np.abs(ph) < win
    x, y = ph[sel], fl[sel]
    if len(x) < 20:
        return None

    d0 = max(float(cand.get("depth", 0.005)) or 0.005, 1e-4)
    h0 = max(durf / 2.0, 2e-3)
    p0 = [0.0, d0, h0, h0 * 0.3, 1.0]
    lo = [-win, 1e-5, 1e-4, 1e-5, 0.9]
    hi = [win, 0.5, win, win, 1.1]
    try:
        popt, pcov = curve_fit(_trapezoid, x, y, p0=p0, bounds=(lo, hi), maxfev=8000)
    except Exception:
        return None
    if not np.all(np.isfinite(pcov)):
        return None
    perr = np.sqrt(np.abs(np.diag(pcov)))
    _, depth, half, _, _ = popt
    _, ddepth, dhalf, _, _ = perr

    dur_hr = 2.0 * half * period * 24.0
    ddur_hr = 2.0 * dhalf * period * 24.0
    p_err = cand.get("period_uncertainty")
    p_err = float(p_err) if (p_err is not None and np.isfinite(p_err)) else None

    def pack(v, e, unit):
        return {"value": round(float(v), 6), "err": (round(float(e), 6) if e is not None else None), "unit": unit}

    return {
        "period": pack(period, p_err, "days"),
        "duration": pack(dur_hr, ddur_hr, "hours"),
        "depth": pack(depth * 1e6, ddepth * 1e6, "ppm"),
    }
