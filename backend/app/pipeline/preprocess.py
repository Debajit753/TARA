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


def clean_lc(lc, window_length: int = 401, sigma: float = 5.0):
    return (lc.remove_nans()
              .normalize()
              .flatten(window_length=window_length)
              .remove_outliers(sigma=sigma))


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
