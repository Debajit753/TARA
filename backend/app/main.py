"""TARA backend — FastAPI app.

/analyze runs the full pipeline on one star and returns a live verdict from the
trained model plus chart data + diagnostics. Fast path:
  - two-stage search (coarse BLS -> narrow TLS refine), periodogram computed once
  - numba/TLS warm-up at startup (first-call JIT compile happens before users)
  - disk cache: every analysed target is saved to backend/data/cache/ and served
    instantly on repeat requests (popular stars are precomputed there).
"""
import os
import re
import json
import math
import time
import threading
import numpy as np
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.preprocess import load_raw, clean_lc
from app.pipeline.search import fast_search, deep_search
from app.pipeline.features import extract_features
from app.pipeline.blend import centroid_shift

app = FastAPI(title="TARA API", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "models", "tabular", "model.joblib")
CACHE_DIR = os.path.join(os.path.dirname(HERE), "data", "cache")
POPULAR_PATH = os.path.join(os.path.dirname(HERE), "data", "popular_stars.json")
os.makedirs(CACHE_DIR, exist_ok=True)
_bundle = None
_ensemble = None


def get_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_ensemble():
    """Lazy-load the multimodal CNN ensemble (torch imported here to keep the
    server's cold start light). Returns None if torch/models are unavailable."""
    global _ensemble
    if _ensemble is None:
        try:
            from app.cnn_infer import Ensemble
            _ensemble = Ensemble(os.path.join(HERE, "models"))
        except Exception:
            _ensemble = False        # sentinel: tried and failed, don't retry
    return _ensemble or None


def _cache_path(tic_id: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", tic_id.strip().upper()).strip("_")
    return os.path.join(CACHE_DIR, f"{key}.json")


# ---------- helpers ----------

def _pair(x, y, n=600):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) > n:
        idx = np.linspace(0, len(x) - 1, n).astype(int); x, y = x[idx], y[idx]
    return [round(float(v), 6) for v in x], [round(float(v), 6) for v in y]


def _diagnostics(f):
    vs = f.get("v_shape", 0) or 0
    sr = f.get("secondary_ratio", 0) or 0
    oe = f.get("odd_even_diff", 0) or 0
    ct = f.get("centroid", 0) or 0
    checks = [
        ("Transit shape (U vs V)", vs >= 0.45, f"{vs:.2f}",
         "Planets carve a flat-bottomed U-dip; grazing binaries make a sharp V.",
         "Flat-bottomed — consistent with a planet.",
         "V-shaped — more like two stars grazing."),
        ("Secondary eclipse", sr < 0.35, f"{sr:.2f}",
         "A second dip half an orbit later means the companion glows — a star, not a planet.",
         "No secondary dip — consistent with a dark planet.",
         "A secondary dip is present — points to an eclipsing binary."),
        ("Odd vs even depth", oe < 0.5, f"{oe:.2f}",
         "If alternating dips differ in depth, it's likely a binary caught at half its true period.",
         "Odd and even dips match — one repeating event.",
         "Alternating dips differ — likely a binary at half period."),
        ("Centroid / blend", ct < 1.0, f"{ct:.2f}σ",
         "If the star's light-centre shifts during the dip, the signal comes from a different nearby star.",
         "Light-centre stays put — the signal is from this star.",
         "Light-centre shifts — a nearby source is blending in."),
    ]
    return [{"label": a, "status": "ok" if ok else "bad", "value": val,
             "tests": tst, "result": (g if ok else b)}
            for a, ok, val, tst, g, b in checks]


def _meta(lc):
    m = getattr(lc, "meta", None) or {}

    def g(*keys):
        for k in keys:
            v = m.get(k)
            if v is not None:
                try:
                    return round(float(v), 3)
                except (TypeError, ValueError):
                    return v
        return None
    return {"ra": g("RA_OBJ", "RA"), "dec": g("DEC_OBJ", "DEC"), "tmag": g("TESSMAG"),
            "teff": g("TEFF"), "radius": g("RADIUS"), "logg": g("LOGG"), "sector": g("SECTOR")}


def _cnn_verdict(lc, cand, feats, meta):
    """Second opinion from the multimodal CNN ensemble. Best-effort: 8 of 11
    scalars come from live measurements + FITS header, 3 (impact/teq/insol) are
    median-filled. Returns None if the ensemble isn't available; never raises."""
    try:
        from app.pipeline.cnn_views import build_views
        from app.cnn_infer import assemble_scalars
        ens = get_ensemble()
        if not ens or not ens.n_models:
            return None
        g, l = build_views(lc, cand)
        scv, n_real = assemble_scalars(feats, meta)
        out = ens.predict(g, l, scv)
        out["scalars_real"] = f"{n_real}/11"
        to = out.get("trained_on", "")
        if "TESS" in to and "Kepler" in to:
            out["note"] = ("deep-learning multimodal verdict — cross-mission model "
                           "(trained on Kepler + TESS), held-out AUC ~0.95.")
        elif to == "TESS":
            out["note"] = ("deep-learning multimodal verdict — TESS fine-tuned "
                           "(held-out TESS AUC ~0.86).")
        else:
            out["note"] = ("deep-learning multimodal verdict — Kepler-trained, not TESS-adapted "
                           "(zero-shot TESS AUC ~0.6); trust the RF + vetting on TESS.")
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- core pipeline (used by the endpoint AND the precompute script) ----------

def run_analysis(tic_id: str, deep: bool = False) -> dict:
    # per-stage wall-clock timings (ms) — shown live in the dashboard's pipeline trace
    tm = {}
    _t0 = time.perf_counter()
    raw = load_raw(tic_id);                    tm["load"] = time.perf_counter() - _t0
    return _analyze_raw(raw, tic_id, tm, _t0, deep=deep)


def _analyze_raw(raw, tic_id, tm, _t0, deep: bool = False) -> dict:
    """Shared pipeline body — used by /analyze (MAST download) and /analyze-file (upload)."""
    _t = time.perf_counter()
    lc = clean_lc(raw);                        tm["clean"] = time.perf_counter() - _t
    _t = time.perf_counter()
    cand, pgdata = fast_search(lc)             # fast two-stage search
    search_note = None
    if deep:
        # thorough full-range TLS (3-min cap) — recovers real periods the fast search
        # aliases on shallow candidates. Keep it only if it beats the fast SDE.
        dcand = deep_search(lc, cap_seconds=180)
        if dcand and dcand.get("power", 0) > cand.get("power", 0) + 0.5:
            search_note = (f"deep search improved the period {cand['period']:.4f} d → "
                           f"{dcand['period']:.4f} d (SDE {cand.get('power',0):.1f} → {dcand['power']:.1f})")
            cand = dcand
        else:
            search_note = "deep search ran — no better period found (or hit the 3-min cap)"
        pgdata["best"] = cand["period"]
    tm["search"] = time.perf_counter() - _t
    _t = time.perf_counter()
    feats, _, _ = extract_features(lc, cand);  tm["features"] = time.perf_counter() - _t
    _t = time.perf_counter()
    feats["centroid"] = centroid_shift(raw, cand); tm["centroid"] = time.perf_counter() - _t

    meta = _meta(raw)
    _t = time.perf_counter()
    cnn = _cnn_verdict(lc, cand, feats, meta); tm["cnn"] = time.perf_counter() - _t
    _t = time.perf_counter()
    try:
        from app.pipeline.fit import fit_transit
        uncertainties = fit_transit(lc, cand)
    except Exception:
        uncertainties = None
    tm["fit"] = time.perf_counter() - _t

    _t = time.perf_counter()
    b = get_model()
    x = np.nan_to_num(np.array([[feats.get(f, 0.0) for f in b["features"]]], dtype=float))
    model = b["model"]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    top = int(np.argmax(proba))
    tm["classify"] = time.perf_counter() - _t

    # --- no-detection guard -------------------------------------------------
    # If the search found no usable transit (flat / noise-dominated curve), the
    # feature vector is degenerate (SNR~0, no in-transit points) and the RF's
    # guess on that all-zeros input isn't meaningful — it tends to default to
    # "false_positive". Report such stars honestly as noise instead.
    snr = feats.get("snr") or 0.0
    depth_f = feats.get("depth") or 0.0
    n_tr = feats.get("n_transits") or 0
    # SNR >= 7.0 mirrors the standard TESS/Kepler detection threshold (~7.1):
    # below it, pipelines don't even report a signal — classifying one is noise-fitting
    detected = (n_tr >= 1) and (depth_f > 0) and (snr >= 7.0)
    # "confident" only if the top class clearly wins — otherwise the model is
    # essentially guessing (near-uniform probs) and a class badge would over-claim.
    _ps = sorted(proba, reverse=True)
    _margin = float(_ps[0] - (_ps[1] if len(_ps) > 1 else 0.0))
    cls = {
        "label": classes[top],
        "confidence": round(float(proba[top]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "detected": bool(detected),
        "confident": bool(detected and _ps[0] >= 0.40 and _margin >= 0.10),
    }
    if not detected:
        cls["label"] = "noise"
        cls["note"] = (f"No transit signal found (SNR≈{snr:.1f}, no clear in-transit "
                       "points) — flat / noise-dominated light curve, reported as noise. "
                       "The vetting checks below need a real signal to be meaningful.")
        # the CNN reads an amplitude-NORMALIZED fold, so with no real dip its input
        # is just stretched noise — its score is structurally meaningless here
        if isinstance(cnn, dict) and "label" in cnn:
            cnn["meaningful"] = False
            cnn["note"] = ("no detected signal to read — the CNN sees an amplitude-"
                           "normalized fold, so without a real dip its input is just "
                           "stretched noise and its score carries no information.")

    lt, lf = _pair(lc.time.value, lc.flux.value, 700)
    pp, ppow = _pair(pgdata["p"], pgdata["power"], 400)
    try:
        fold = lc.fold(period=cand["period"], epoch_time=cand["t0"], normalize_phase=True)
        pha = np.asarray(fold.phase.value, float)
        fla = np.asarray(fold.flux.value, float)
        g = np.isfinite(pha) & np.isfinite(fla)
        pha, fla = pha[g], fla[g]
        # Scatter for the chart: keep EVERY point near the transit (the zoomed window
        # the UI shows), and only thin the far-off baseline. This gives the dense,
        # real folded curve instead of a sparse downsample across the whole phase.
        near = np.abs(pha) <= 0.30
        i_near = np.where(near)[0]
        i_far = np.where(~near)[0]
        if len(i_near) > 6000:                     # generous cap so canvas stays snappy
            i_near = i_near[np.linspace(0, len(i_near) - 1, 6000).astype(int)]
        if len(i_far) > 800:                       # baseline is just context — thin it
            i_far = i_far[np.linspace(0, len(i_far) - 1, 800).astype(int)]
        keep = np.sort(np.concatenate([i_near, i_far]))
        fph = [round(float(v), 6) for v in pha[keep]]
        ff = [round(float(v), 6) for v in fla[keep]]
        # binned medians from the FULL fold (not the scatter above)
        nb = 240
        idxb = np.clip(np.digitize(pha, np.linspace(-0.5, 0.5, nb + 1)) - 1, 0, nb - 1)
        bph, bf = [], []
        for b in range(nb):
            v = fla[idxb == b]
            if len(v) >= 3:
                bph.append(round(float(-0.5 + (b + 0.5) / nb), 4))
                bf.append(round(float(np.median(v)), 6))
    except Exception:
        fph, ff, bph, bf = [], [], [], []
    period = cand["period"]; durf = (cand["duration"] / period) if period else 0.05
    depth = feats.get("depth", 0) or 0
    mph = np.linspace(-0.5, 0.5, 220)

    def trap(ph):
        a = abs(ph); half = durf / 2; flat = half * 0.6
        if a <= flat: return 1 - depth
        if a <= half: return 1 - depth * (1 - (a - flat) / (half - flat + 1e-9))
        return 1.0
    mf = [round(float(trap(p)), 6) for p in mph]

    def rnd(v):
        return round(float(v), 5) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v

    tm["total"] = time.perf_counter() - _t0
    return {
        "tic_id": tic_id,
        "target": meta,
        "timings": {k: round(v * 1000.0, 1) for k, v in tm.items()},   # ms
        "search_note": search_note,
        "classification": cls,
        "cnn": cnn,
        "parameters": {k: rnd(v) for k, v in feats.items()},
        "uncertainties": uncertainties,
        "diagnostics": _diagnostics(feats),
        "charts": {
            "light_curve": {"t": lt, "f": lf},
            "periodogram": {"p": pp, "power": ppow, "best": round(float(period), 4)},
            "fold": {"ph": fph, "f": ff, "bph": bph, "bf": bf,
                     "model_ph": [round(float(p), 4) for p in mph], "model_f": mf},
        },
    }


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH),
            "cached_targets": len(os.listdir(CACHE_DIR))}


@app.get("/popular")
def popular():
    if os.path.exists(POPULAR_PATH):
        with open(POPULAR_PATH) as f:
            stars = json.load(f)
        for s in stars:   # tell the UI which stars will answer instantly
            s["cached"] = os.path.exists(_cache_path(s.get("tic", "")))
        return stars
    return []


def _finite(o):
    """Replace non-finite floats (inf/nan) with None so the JSON response is valid
    (Starlette serialises with allow_nan=False, which would 500 on inf/nan)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


@app.post("/analyze")
def analyze(tic_id: str, refresh: bool = False, deep: bool = False):
    cp = _cache_path(tic_id)
    if not refresh and not deep and os.path.exists(cp):
        try:
            with open(cp) as f:
                out = json.load(f)
            out["cached"] = True
            return _finite(out)
        except (ValueError, OSError):
            pass          # corrupt/truncated cache -> fall through and recompute (self-heals)
    try:
        out = _finite(run_analysis(tic_id, deep=deep))   # deep = thorough 3-min search
    except Exception as e:
        return {"tic_id": tic_id, "error": f"{type(e).__name__}: {e}"}
    # atomic write: a crash mid-write can never leave a truncated file that 500s forever
    tmp = cp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, cp)
    return out


@app.post("/analyze-file")
async def analyze_file(request: Request, name: str = "upload.fits"):
    """Analyze an UPLOADED light-curve FITS file (files instead of TIC IDs).
    Raw body = the file bytes; same pipeline as /analyze, just skipping the
    MAST download."""
    data = await request.body()
    if not data:
        return {"error": "empty file"}
    updir = os.path.join(os.path.dirname(HERE), "data", "uploads")
    os.makedirs(updir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "upload.fits"
    path = os.path.join(updir, safe)
    with open(path, "wb") as f:
        f.write(data)
    try:
        import lightkurve as lk
        tm = {}
        _t0 = time.perf_counter()
        raw = lk.read(path)
        if hasattr(raw, "to_lightcurve"):        # a target-pixel file, not a light curve
            try:
                raw = raw.to_lightcurve()
            except Exception:
                raw = raw.to_lightcurve(aperture_mask="threshold")
        tm["load"] = time.perf_counter() - _t0
        return _finite(_analyze_raw(raw, safe, tm, _t0))
    except Exception as e:
        return {"tic_id": safe, "error": f"{type(e).__name__}: {e}"}


@app.on_event("startup")
def _warmup():
    """Trigger numba/TLS JIT compilation in the background so the first real
    request doesn't pay the ~30-60 s compile cost."""
    def go():
        try:
            from transitleastsquares import transitleastsquares
            rng = np.random.default_rng(0)
            t = np.linspace(0.0, 6.0, 900)
            f = 1.0 + 1e-4 * rng.standard_normal(900)
            transitleastsquares(t, f).power(
                period_min=1.0, period_max=1.5, use_threads=2,
                oversampling_factor=1, duration_grid_step=1.2,
                show_progress_bar=False)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()
