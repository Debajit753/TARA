"""Stage 3 — extract features from a cleaned light curve + a BLS candidate.

Produces the inputs the classifier uses:
  - shape parameters (depth, duration, duration/period)
  - vetting diagnostics (odd-even depth diff, secondary-eclipse ratio, V/U shape)
  - signal strength (SNR)
  - global + local binned views (for the 1D-CNN, later)
All numbers come from numpy — no heavy dependencies needed here.
"""
import numpy as np


def _phase(time, period, t0):
    """Phase in [-0.5, 0.5], transit centred at 0."""
    return ((time - t0 + 0.5 * period) % period) / period - 0.5


def _binned(ph, flux, half_width, nbins):
    """Median-binned view of flux vs phase within +/- half_width."""
    sel = np.abs(ph) < half_width
    x, y = ph[sel], flux[sel]
    out = np.ones(nbins)
    if len(y) == 0:
        return out
    bins = np.linspace(-half_width, half_width, nbins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)
    for b in range(nbins):
        vals = y[idx == b]
        if len(vals):
            out[b] = np.median(vals)
    return out


def extract_features(lc, candidate, n_global=201, n_local=61):
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]

    period = float(candidate["period"])
    t0 = float(candidate["t0"])
    dur = float(candidate["duration"])
    dur_frac = dur / period if period > 0 else 0.05

    ph = _phase(time, period, t0)
    in_tr = np.abs(ph) < 0.5 * dur_frac
    oot = np.abs(ph) > 1.5 * dur_frac

    sigma = np.std(flux[oot]) if oot.sum() > 10 else np.std(flux)
    meas_depth = (1.0 - np.median(flux[in_tr])) if in_tr.sum() > 3 else float(candidate.get("depth", 0))
    snr = meas_depth / (sigma / np.sqrt(max(in_tr.sum(), 1)) + 1e-12)

    # odd vs even transit depth
    epoch = np.round((time - t0) / period) if period > 0 else np.zeros_like(time)
    odd = in_tr & (epoch % 2 == 1)
    even = in_tr & (epoch % 2 == 0)
    # NOTE on the `else` fallbacks below: when a test has too few points it falls
    # back to a value that happens to read as INNOCENT (odd==even -> diff 0, "dips
    # match"). The model was trained with these same fallbacks, so they are kept
    # verbatim to avoid train/serve skew — but each test now also reports whether
    # it was actually measurable, so the UI can say "not testable" instead of
    # printing a clean pass for a check that never ran.
    ok_odd_even = bool(odd.sum() > 3 and even.sum() > 3)
    d_odd = (1 - np.median(flux[odd])) if odd.sum() > 3 else meas_depth
    d_even = (1 - np.median(flux[even])) if even.sum() > 3 else meas_depth
    odd_even_diff = abs(d_odd - d_even) / (meas_depth + 1e-9)

    # secondary eclipse near phase 0.5
    sec_mask = (np.abs(ph - 0.5) < 0.5 * dur_frac) | (np.abs(ph + 0.5) < 0.5 * dur_frac)
    ok_secondary = bool(sec_mask.sum() > 3)
    sec_depth = (1 - np.median(flux[sec_mask])) if sec_mask.sum() > 3 else 0.0
    sec_ratio = max(sec_depth, 0.0) / (meas_depth + 1e-9)

    # V-vs-U shape: flat bottom (core) vs sloped wings
    core = np.abs(ph) < 0.25 * dur_frac
    wing = (np.abs(ph) > 0.35 * dur_frac) & (np.abs(ph) < 0.5 * dur_frac)
    ok_v_shape = bool(core.sum() > 3 and wing.sum() > 3)
    d_core = (1 - np.median(flux[core])) if core.sum() > 3 else meas_depth
    d_wing = (1 - np.median(flux[wing])) if wing.sum() > 3 else meas_depth * 0.5
    v_shape = d_wing / (d_core + 1e-9)  # ~1 = U (flat, planet-like), smaller = V

    features = {
        "period": period,
        "depth": float(meas_depth),
        "duration": dur,
        "duration_frac": float(dur_frac),
        "snr": float(snr),
        "odd_even_diff": float(odd_even_diff),
        "secondary_ratio": float(sec_ratio),
        "v_shape": float(v_shape),
        "n_transits": int(np.unique(epoch[in_tr]).size),
        # variability features (v3): describe the REST of the curve, not the dip —
        # a variable star that phases into a clean-looking dip still has a messy baseline
        "oot_scatter": float(sigma),
        "p2p_rms": float(np.sqrt(np.mean(np.diff(flux) ** 2))) if len(flux) > 2 else 0.0,
        # --- testability flags (UI only; NOT model inputs — the model reads the
        # fixed 12-name list in model.joblib["features"], so extra keys are ignored)
        "testable": {
            "odd_even_diff": ok_odd_even,
            "secondary_ratio": ok_secondary,
            "v_shape": ok_v_shape,
        },
        "n_points": {
            "odd": int(odd.sum()), "even": int(even.sum()),
            "secondary": int(sec_mask.sum()),
            "core": int(core.sum()), "wing": int(wing.sum()),
        },
    }
    global_view = _binned(ph, flux, 0.5, n_global)
    local_view = _binned(ph, flux, max(2 * dur_frac, 0.05), n_local)
    return features, global_view, local_view
