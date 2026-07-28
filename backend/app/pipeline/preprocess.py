"""Stage 1 — load & clean the light curve.

load_raw keeps the raw light curve (including its centroid columns, used by the
blend check). clean_lc detrends it for the transit search."""
import os
import re
import time
import numpy as np
import lightkurve as lk


def load_raw(tic_id: str, retries: int = 3):
    """Fetch the raw light curve; retry once on transient MAST network errors."""
    # normalize the input so a bare number, "TIC123", "tic 123" all resolve
    q = tic_id.strip()
    d = q[3:].strip() if q.upper().startswith("TIC") else q
    if d.isdigit():
        q = f"TIC {d}"
    elif not any(c.isdigit() for c in q):
        raise ValueError(f"Enter a TIC number, e.g. 'TIC 219253008' (you entered '{tic_id.strip()}')")
    last_err = None
    had_data = False

    def _drop_corrupt(err):
        # a corrupt/interrupted download leaves a bad .fits that keeps failing to read —
        # delete it so the next attempt (or the FFI fallback) re-fetches a fresh copy
        m = re.search(r"(/\S+\.fits)", str(err))
        if m and os.path.exists(m.group(1)):
            try:
                os.remove(m.group(1))
            except OSError:
                pass

    # 1) pre-made light curves, preferring reliable pipelines, with corrupt-file recovery
    for attempt in range(retries):
        try:
            search = lk.search_lightcurve(q, mission="TESS")
            if len(search) == 0:
                # MAST sometimes returns an EMPTY table transiently for a star that
                # has plenty of data — retry before concluding "no products"
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                break                                    # consistently empty -> try the FFI below
            had_data = True
            authors = np.asarray(search.table["author"])
            pick = None
            for auth in ("SPOC", "TESS-SPOC", "QLP", "CDIPS", "GSFC-ELEANOR-LITE"):
                idx = np.where(authors == auth)[0]
                if len(idx):
                    pick = search[int(idx[0])]
                    break
            # NOTE: uses a single (the first) sector. Stitching all sectors was tested
            # and did NOT help hard shallow candidates (e.g. TOI-1022) — the limit is
            # the fast two-stage SEARCH locking a period alias, not the amount of data;
            # a full TLS search would help but is too slow live on long baselines.
            return (pick if pick is not None else search[0]).download()
        except Exception as e:
            last_err = e
            _drop_corrupt(e)
            time.sleep(1.0 * (attempt + 1))

    # 2) fall back to extracting from the Full-Frame Images (no products, OR the pre-made
    #    products are all flaky/corrupt — e.g. TARS-only stars). Slower & noisier but works.
    try:
        cut = lk.search_tesscut(q)
        if len(cut):
            had_data = True
            tpf = cut[0].download(cutout_size=11)
            return tpf.to_lightcurve(aperture_mask="threshold")
    except Exception as e:
        last_err = e
        _drop_corrupt(e)

    if not had_data:
        # honest ambiguity: an empty MAST answer can mean "never observed" OR
        # "the archive didn't respond just now" — say both, suggest a retry
        raise ValueError(f"No TESS data came back for {tic_id} — either it was never "
                         "observed, or the NASA archive didn't respond; try again in a moment.")
    raise ValueError(f"Could not read TESS data for {tic_id} — the download was unreadable, try again in a moment.")


def clean_lc(lc, window_length: int = None, sigma: float = 5.0,
             window_days: float = 0.55):
    # zero-centered flux guard: some products deliver flux with median ~0 —
    # normalize() would divide by ~0 and every measurement becomes garbage.
    # Scale is measured with the MAD (robust): plain nanstd is inflated by
    # scattered-light spikes, which made this guard fire on good FFI products.
    f0 = np.asarray(lc.flux.value, float)
    m0 = np.nanmedian(f0)
    mad = 1.4826 * np.nanmedian(np.abs(f0 - m0)) if np.isfinite(m0) else np.nan
    if (not np.isfinite(m0)) or m0 <= 0:
        raise ValueError("This star's light-curve product has zero-centered flux "
                         "(median <= 0), which is unusable for relative photometry. "
                         "This product cannot be analysed; the star may need FFI extraction.")
    if np.isfinite(mad) and m0 < 3.0 * mad:
        raise ValueError("This star's light curve is dominated by scatter rather than "
                         "signal (median flux is below 3x the noise scale) — no reliable "
                         "relative photometry is possible from this product.")

    lc = lc.remove_nans().normalize()

    # The savgol window is a CADENCE COUNT, so a fixed 401 means ~13 h on 2-min
    # data but ~8.4 DAYS on 30-min FFI data — i.e. effectively no detrending at
    # all there. Derive it from real time instead. window_days=0.55 reproduces
    # the historical 401 on 2-min data, so the well-tested SPOC path is unchanged.
    if window_length is None:
        t = np.asarray(lc.time.value, float)
        dt = np.nanmedian(np.diff(t)) if t.size > 2 else np.nan
        if np.isfinite(dt) and dt > 0:
            window_length = max(11, int(round(window_days / dt)) | 1)   # odd, >= 11
        else:
            window_length = 401
    n = int(np.isfinite(np.asarray(lc.flux.value, float)).sum())
    if window_length >= n:                       # savgol needs window < len(data)
        window_length = max(5, (n // 2) | 1)

    # NOTE sigma_lower=inf: remove_outliers is SYMMETRIC by default, and a transit
    # IS a run of low outliers — a deep, low-duty-cycle transit was being deleted
    # entirely by its own cleaning step. Clip flares/cosmic rays (high side) only.
    return (lc.flatten(window_length=window_length)
              .remove_outliers(sigma=sigma, sigma_lower=np.inf))


def load_and_clean(tic_id: str, window_length: int = 401, sigma: float = 5.0):
    return clean_lc(load_raw(tic_id), window_length, sigma)


def clean_lc_biweight(lc, window_days: float = 0.4, sigma: float = 5.0, mask=None):
    """Robust detrending with wotan's biweight estimator instead of Savitzky-Golay.

    Savgol `flatten()` fits a smooth polynomial through EVERY point, so a deep or
    long transit gets partly absorbed into the trend (the dip is suppressed).
    wotan's biweight is outlier-robust — it treats the in-transit points as
    outliers to the stellar trend and leaves the dip intact. Optionally pass
    `mask` (bool array, True = in-transit) for a proper 2-pass detrend that
    ignores the known transit entirely while fitting the trend."""
    import lightkurve as lk
    from wotan import flatten as wotan_flatten
    lc = lc.remove_nans().normalize()
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    flat = wotan_flatten(t, f, window_length=window_days, method="biweight",
                         mask=mask, return_trend=False)
    good = np.isfinite(flat)
    out = lk.LightCurve(time=t[good], flux=flat[good])
    return out.remove_outliers(sigma=sigma)
