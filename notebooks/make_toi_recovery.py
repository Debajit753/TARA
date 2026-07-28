"""Generates TWO notebooks that measure TARA's REAL recovery on known TOIs:
  tara_colab_toi_recovery.ipynb   (Colab, PART 1 & 2)
  tara_kaggle_toi_recovery.ipynb  (Kaggle, PART 3 & 4)

4 accounts run a quarter each. Each runs TARA's own pipeline (its live BLS->TLS
search + the 4-class RandomForest verdict, with the SNR>=7 no-detection guard and
the confident/uncertain logic) on a balanced sample of TOIs, and scores the verdict
against the catalog disposition. Then a merge cell builds the recovery table:
how many known planets recovered as 'transit', how many EBs / false-positives caught.

Uploadable (NO API fetch): upload model.joblib + the TOI catalog CSV.
RandomForest-only (TARA's primary verdict) — keeps it light; the CNN second
opinion is not included."""
import json

# ---------------------------------------------------------------- shared code
CONFIG = '''# ---- config: set PART per account -------------------------------------------
PART = 1            # Colab acct A=1, acct B=2   |   Kaggle acct C=3, acct D=4
N_PARTS = 4
# how many TOIs to sample per disposition (balanced). ~330 total -> ~82 per account.
PER = {"CP": 60, "KP": 60, "PC": 80, "EB": 80, "FP": 1, "O": 20, "IS": 10, "V": 20}
TIMEOUT_S = 120     # per-star watchdog'''

SAMPLE = '''# ---- parse the TOI catalog + balanced sample, split by PART ------------------
import numpy as np, pandas as pd
df = pd.read_csv(CSV_PATH, comment="#")
df = df.dropna(subset=["TIC", "TOI Disposition"])
df["TIC"] = df["TIC"].astype(int)
df = df.drop_duplicates(subset="TIC")
print("catalog dispositions:", df["TOI Disposition"].value_counts().to_dict())

# what TARA SHOULD say for each disposition
EXPECT = {"CP": "planet", "KP": "planet", "PC": "planet",
          "EB": "eclipsing_binary", "FP": "not_planet", "O": "not_planet",
          "IS": "not_planet", "V": "not_planet"}
parts = []
for disp, n in PER.items():
    sub = df[df["TOI Disposition"] == disp]
    if len(sub): parts.append(sub.sample(min(n, len(sub)), random_state=42))
full = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
samp = full.iloc[(PART-1)::N_PARTS].reset_index(drop=True)
print(f"PART {PART}/{N_PARTS}: {len(samp)} TOIs",
      samp["TOI Disposition"].value_counts().to_dict())'''

PIPE = '''# ---- TARA's pipeline, embedded (matches the live backend) --------------------
import numpy as np, lightkurve as lk, signal, time, joblib
from transitleastsquares import transitleastsquares
BUNDLE = joblib.load(MODEL_PATH); MODEL = BUNDLE["model"]; FEATS = BUNDLE["features"]
print("model classes:", list(MODEL.classes_), "| features:", len(FEATS))

class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(_TO()))

def clean(lcr):
    f0 = np.asarray(lcr.flux.value, float); m0 = np.nanmedian(f0)
    if (not np.isfinite(m0)) or m0 <= 0 or m0 < np.nanstd(f0):
        raise ValueError("zero-centered flux")
    return lcr.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)

def fast_search(lc, minp=0.5, maxp=10.0):
    try: blc = lc.bin(time_bin_size=10.0/1440.0)
    except Exception: blc = lc
    pg = blc.to_periodogram(method="bls", minimum_period=minp, maximum_period=maxp, frequency_factor=300)
    p0 = float(pg.period_at_max_power.value)
    lo, hi = max(minp, p0*0.97), min(maxp*1.2, p0*1.03)
    t = np.ascontiguousarray(lc.time.value, float); f = np.ascontiguousarray(lc.flux.value, float)
    m = np.isfinite(t) & np.isfinite(f)
    try:
        r = transitleastsquares(t[m], f[m]).power(period_min=lo, period_max=hi,
              oversampling_factor=2, duration_grid_step=1.15, use_threads=2, show_progress_bar=False)
        return {"period": float(r.period), "t0": float(r.T0), "depth": float(abs(1-r.depth)), "duration": float(r.duration)}
    except Exception:
        return {"period": p0, "t0": float(pg.transit_time_at_max_power.value),
                "depth": float(pg.depth_at_max_power), "duration": float(pg.duration_at_max_power.value)}

def features(lc, lcr, c):
    t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float)
    g = np.isfinite(t) & np.isfinite(f); t, f = t[g], f[g]
    P, t0, dur = float(c["period"]), float(c["t0"]), float(c["duration"])
    durf = dur/P if P > 0 else 0.05
    ph = ((t - t0 + 0.5*P) % P)/P - 0.5
    intr = np.abs(ph) < 0.5*durf; oot = np.abs(ph) > 1.5*durf
    sigma = np.std(f[oot]) if oot.sum() > 10 else np.std(f)
    depth = (1-np.median(f[intr])) if intr.sum() > 3 else float(c.get("depth", 0))
    snr = depth/(sigma/np.sqrt(max(intr.sum(),1)) + 1e-12)
    ep = np.round((t-t0)/P) if P > 0 else np.zeros_like(t)
    odd, even = intr & (ep % 2 == 1), intr & (ep % 2 == 0)
    d_o = (1-np.median(f[odd])) if odd.sum() > 3 else depth
    d_e = (1-np.median(f[even])) if even.sum() > 3 else depth
    oe = abs(d_o-d_e)/(depth+1e-9)
    sec = (np.abs(ph-0.5) < 0.5*durf) | (np.abs(ph+0.5) < 0.5*durf)
    sd = (1-np.median(f[sec])) if sec.sum() > 3 else 0.0
    sr = max(sd, 0.0)/(depth+1e-9)
    core = np.abs(ph) < 0.25*durf; wing = (np.abs(ph) > 0.35*durf) & (np.abs(ph) < 0.5*durf)
    d_c = (1-np.median(f[core])) if core.sum() > 3 else depth
    d_w = (1-np.median(f[wing])) if wing.sum() > 3 else depth*0.5
    vs = d_w/(d_c+1e-9)
    def col(*names):
        for n in names:
            if n in lcr.columns: return np.asarray(lcr[n].value, float)
        return None
    c1, c2 = col("centroid_col","mom_centr1","pos_corr1"), col("centroid_row","mom_centr2","pos_corr2")
    cen = np.nan
    if c1 is not None and c2 is not None:
        tr = np.asarray(lcr.time.value, float); phr = ((tr-t0+0.5*P) % P)/P - 0.5
        ri, ro = np.abs(phr) < 0.5*durf, np.abs(phr) > 1.5*durf
        gg = np.isfinite(c1) & np.isfinite(c2)
        if (ri&gg).sum() >= 3 and (ro&gg).sum() >= 3:
            scv = np.hypot(np.std(c1[ro&gg]), np.std(c2[ro&gg])) + 1e-9
            sh = np.hypot(np.median(c1[ri&gg])-np.median(c1[ro&gg]), np.median(c2[ri&gg])-np.median(c2[ro&gg]))
            cen = float(sh/scv)
    return {"period": P, "depth": float(depth), "duration": dur, "duration_frac": float(durf),
            "snr": float(snr), "odd_even_diff": float(oe), "secondary_ratio": float(sr),
            "v_shape": float(vs), "n_transits": int(np.unique(ep[intr]).size), "centroid": cen,
            "oot_scatter": float(sigma),
            "p2p_rms": float(np.sqrt(np.mean(np.diff(f)**2))) if len(f) > 2 else 0.0}

def classify(feats):
    x = np.nan_to_num(np.array([[feats.get(k, 0.0) for k in FEATS]], dtype=float))
    proba = MODEL.predict_proba(x)[0]; classes = list(MODEL.classes_); top = int(np.argmax(proba))
    snr = feats.get("snr") or 0; depth = feats.get("depth") or 0; ntr = feats.get("n_transits") or 0
    detected = (ntr >= 1) and (depth > 0) and (snr >= 7.0)         # TARA no-detection guard
    ps = sorted(proba, reverse=True); margin = ps[0]-(ps[1] if len(ps) > 1 else 0)
    return {"verdict": classes[top] if detected else "noise",
            "confidence": round(float(proba[top]), 3), "detected": bool(detected),
            "confident": bool(detected and ps[0] >= 0.40 and margin >= 0.10)}

def analyze(tic):
    sr = lk.search_lightcurve(f"TIC {int(tic)}", mission="TESS", author="SPOC")
    if len(sr) == 0: sr = lk.search_lightcurve(f"TIC {int(tic)}", mission="TESS")
    if len(sr) == 0: raise ValueError("no LC")
    lcr = sr[0].download(); lc = clean(lcr)
    cand = fast_search(lc); feats = features(lc, lcr, cand)
    out = classify(feats); out["period"] = round(feats["period"], 4); return out'''

RUN = '''# ---- run the loop, save incrementally ---------------------------------------
import time, signal, pandas as pd
rows = []
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT).tic.astype(int)); print(f"resuming — {len(done)} done")
t_start = time.time()
for i, r in samp.iterrows():
    tic = int(r["TIC"]);  disp = r["TOI Disposition"]
    if tic in done: continue
    signal.alarm(TIMEOUT_S); t0 = time.time()
    try:
        a = analyze(tic)
        row = {"tic": tic, "disp": disp, "expect": EXPECT.get(disp, "?"),
               "verdict": a["verdict"], "confidence": a["confidence"],
               "detected": a["detected"], "confident": a["confident"],
               "tara_period": a["period"], "catalog_period": r.get("Orbital Period (days) Value"),
               "secs": round(time.time()-t0, 1)}
    except (Exception, _TO) as e:
        row = {"tic": tic, "disp": disp, "expect": EXPECT.get(disp, "?"),
               "verdict": "FAIL:"+type(e).__name__, "secs": round(time.time()-t0, 1)}
    finally:
        signal.alarm(0)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False);
    if (i+1) % 5 == 0 or i == len(samp)-1:
        print(f"[{i+1}/{len(samp)}] TIC {tic} ({disp}) -> {row['verdict']}  ({(time.time()-t_start)/60:.0f} min)")
print("DONE ->", OUT)'''

MERGE_MD = '''## Recovery table — run once ALL FOUR parts are collected
Upload the other three `toi_recovery_p*.csv` files, then run. Scores TARA's verdict
against the catalog disposition: how many known planets it recovered as `transit`,
how many eclipsing binaries and false positives it correctly caught.'''

MERGE = '''import glob, pandas as pd
fs = sorted(glob.glob("toi_recovery_p*.csv")) + sorted(glob.glob(INPUT_GLOB))
d = pd.concat([pd.read_csv(f) for f in set(fs)]).drop_duplicates(subset="tic").reset_index(drop=True)
ok = d[~d.verdict.astype(str).str.startswith("FAIL")]
print(f"{len(d)} TOIs, {len(d)-len(ok)} failed to load\\n")
print("=== TARA verdict by catalog disposition ===")
print(pd.crosstab(ok["disp"], ok["verdict"]))
# recovery scores
pl = ok[ok.expect == "planet"]; eb = ok[ok.expect == "eclipsing_binary"]; fp = ok[ok.expect == "not_planet"]
print("\\n=== RECOVERY ===")
if len(pl): print(f"planets (CP/KP/PC): {(pl.verdict=='transit').sum()}/{len(pl)} recovered as 'transit' "
                  f"({100*(pl.verdict=='transit').mean():.0f}%) · not-noise: {(pl.verdict!='noise').sum()}/{len(pl)}")
if len(eb): print(f"eclipsing binaries: {(eb.verdict=='eclipsing_binary').sum()}/{len(eb)} caught as EB "
                  f"({100*(eb.verdict=='eclipsing_binary').mean():.0f}%)")
if len(fp): print(f"false pos / other : {(fp.verdict!='transit').sum()}/{len(fp)} correctly NOT 'transit' "
                  f"({100*(fp.verdict!='transit').mean():.0f}%)")
print("\\nHONEST NOTE: single-sector, TARA's own live search, RF verdict only. This is the")
print("real out-of-distribution recovery — expect it well below the 0.95 in-distribution AUC.")'''


def build(platform):
    C = []
    def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": s})
    def code(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

    if platform == "colab":
        md('''# TARA — TOI recovery test (Colab · accounts 1 & 2)

Runs TARA's own pipeline on known TOIs and scores recovery vs the catalog
disposition. **Set `PART = 1` (account A) or `2` (account B), then Run all.**
Uploads: `model.joblib` (from `tara/backend/app/models/tabular/`) + the TOI catalog CSV.
No API fetch. CPU runtime is fine.''')
        code(CONFIG)
        code('''!pip install -q lightkurve transitleastsquares''')
        code('''# ---- upload model.joblib + the TOI catalog CSV ----
import os
from google.colab import files
need = {"model.joblib": None, "csv": None}
for f in os.listdir("."):
    if f == "model.joblib": need["model.joblib"] = f
    if f.endswith(".csv") and "toi" in f.lower(): need["csv"] = f
if not need["model.joblib"] or not need["csv"]:
    print("Upload model.joblib AND the toi-catalog CSV:")
    up = files.upload()
    for k in up:
        if k == "model.joblib": need["model.joblib"] = k
        if k.endswith(".csv"): need["csv"] = k
MODEL_PATH = need["model.joblib"]; CSV_PATH = need["csv"]
OUT = f"toi_recovery_p{PART}.csv"; INPUT_GLOB = "toi_recovery_p*.csv"
print("model:", MODEL_PATH, "| csv:", CSV_PATH)''')
        code(SAMPLE)
        code(PIPE)
        code(RUN + '\nfrom google.colab import files as _f; _f.download(OUT)')
        md(MERGE_MD)
        code(MERGE)
        out = "tara_colab_toi_recovery.ipynb"
    else:  # kaggle
        md('''# TARA — TOI recovery test (Kaggle · accounts 3 & 4)

Runs TARA's own pipeline on known TOIs and scores recovery vs the catalog
disposition. **Settings → Internet ON.** Upload `model.joblib` + the TOI catalog
CSV via **+ Add Input → Upload**. Set `PART = 3` (account C) or `4` (account D),
then Run all. Output `toi_recovery_p{PART}.csv` appears in the Output panel.''')
        code(CONFIG)
        code('''!pip install -q lightkurve transitleastsquares''')
        code('''# ---- find uploaded model.joblib + TOI CSV (via + Add Input -> Upload) ----
import os, glob
def _find(pred):
    for root,_,fs in os.walk("/kaggle/input"):
        for f in fs:
            if pred(f): return os.path.join(root, f)
    for f in os.listdir("."):
        if pred(f): return f
    return None
MODEL_PATH = _find(lambda f: f == "model.joblib")
CSV_PATH   = _find(lambda f: f.endswith(".csv") and "toi" in f.lower())
assert MODEL_PATH and CSV_PATH, "attach model.joblib and the toi-catalog CSV via '+ Add Input -> Upload'"
OUT = f"toi_recovery_p{PART}.csv"; INPUT_GLOB = "/kaggle/input/**/toi_recovery_p*.csv"
print("model:", MODEL_PATH, "| csv:", CSV_PATH)''')
        code(SAMPLE)
        code(PIPE)
        code(RUN + '\nprint("download toi_recovery_p{}.csv from the Output panel".format(PART))')
        md(MERGE_MD.replace("Upload the other three", "Attach the other three (via + Add Input)"))
        code(MERGE)
        out = "tara_kaggle_toi_recovery.ipynb"

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "colab": {"provenance": []}}, "cells": C}
    path = __file__.replace("make_toi_recovery.py", out)
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", path)


build("colab")
build("kaggle")
