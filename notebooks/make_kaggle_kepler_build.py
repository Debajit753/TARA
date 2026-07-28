"""Generates tara_kaggle_kepler_build.ipynb — the KEPLER half of the RF4-v3 build
(parts 1 & 2) plus the final merge/train, as a KAGGLE notebook.

Kaggle differences vs Colab, all handled:
- no google.colab upload/download popups -> input files are attached via
  "+ Add Input -> Upload" (they appear under /kaggle/input/<dataset>/), and
  outputs are picked up from the notebook's Output panel (/kaggle/working)
- Internet must be switched ON in the notebook settings (needed for the MAST
  light-curve downloads; requires a phone-verified account)
- SIGALRM watchdog works (Linux) and sessions allow multi-hour CPU runs

Run TWO sessions in parallel: this notebook with PART=1, and a "Copy & Edit"
duplicate with PART=2. Same 12-feature pipeline as the Colab v3 notebook —
outputs are byte-compatible (rf4v3k_features_p{PART}.csv)."""
import json

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — Kepler RF4-v3 build on **Kaggle** (parts 1 & 2 + merge/train)

The Kepler half of the 4-class RandomForest v3 build (blends/EB/noise from the
DR25 robovetter flags), 12 features, identical to the Colab notebook — just
Kaggle-native.

## Setup (once per session)
1. **Settings (right panel) → Internet → ON** — the light curves download from
   NASA MAST. (Needs a phone-verified Kaggle account.)
2. Download the KOI catalog **in your browser**, rename it `koi_dr25.csv`:
   `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,koi_disposition,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_time0bk,koi_duration+from+cumulative&format=csv`
3. **+ Add Input → Upload** → drag `koi_dr25.csv` in (any dataset name).
4. Set `PART` below → **Run All**.

## Two parts in parallel
Run this notebook with `PART = 1`, then **Copy & Edit** and run the copy with
`PART = 2` at the same time (~1.5–2 h each, resumable within a session).

## Where the output appears
`rf4v3k_features_p{PART}.csv` lands in `/kaggle/working` — download it from the
**Output** panel on the right (use *Save Version* to snapshot it permanently).''')

code('''PART = 1            # <-- 1 in this session, 2 in the Copy & Edit session
N_PARTS = 2
PER_CLASS_K = 500    # per class; blend takes up to 2x (~1000) — it's the point
TIMEOUT_S = 120      # per-star watchdog
!pip install -q lightkurve''')

code('''# ---- find the UPLOADED koi_dr25.csv (attached via "+ Add Input -> Upload") ----
import numpy as np, pandas as pd, os
kcsv = None
for root, _, fs in os.walk("/kaggle/input"):
    for f in fs:
        if f.endswith(".csv") and "koi" in f.lower():
            kcsv = os.path.join(root, f)
if kcsv is None:                       # also allow a copy dropped straight into working
    kcsv = next((f for f in os.listdir(".") if f.endswith(".csv") and "koi" in f.lower()), None)
if kcsv is None:
    raise SystemExit("koi_dr25.csv not found — attach it via '+ Add Input -> Upload' "
                     "(download link is in the cell above), then Run All again.")
print("using catalog:", kcsv)

kd = pd.read_csv(kcsv, comment="#").dropna(subset=["koi_period","koi_time0bk","koi_duration"])
disp = kd.koi_disposition.str.upper()
is_fp = disp == "FALSE POSITIVE"
kd["cls"] = np.where(disp.isin(["CONFIRMED","CANDIDATE"]), "transit",
            np.where(is_fp & ((kd.koi_fpflag_co==1)|(kd.koi_fpflag_ec==1)), "blend",
            np.where(is_fp & (kd.koi_fpflag_ss==1), "eclipsing_binary",
            np.where(is_fp & (kd.koi_fpflag_nt==1), "noise", None))))
kd = kd.dropna(subset=["cls"]).drop_duplicates(subset=["kepid","cls"]).reset_index(drop=True)
print("Kepler class counts:", kd.cls.value_counts().to_dict())
kparts = []
for c, grp in kd.groupby("cls"):
    cap = PER_CLASS_K*2 if c == "blend" else PER_CLASS_K
    kparts.append(grp.sample(n=min(cap, len(grp)), random_state=42))
kfull = pd.concat(kparts).sample(frac=1, random_state=42).reset_index(drop=True)
ksamp = kfull.iloc[(PART-1)::N_PARTS].reset_index(drop=True)
print(f"Kepler part {PART}/{N_PARTS}: building {len(ksamp)}", ksamp.cls.value_counts().to_dict())''')

code('''# ---- the EXACT live-pipeline features, v3 = 12 (mirrors backend features.py + blend.py) ----
import lightkurve as lk, signal, time
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(_TO()))

def _phase(t, p, t0): return ((t - t0 + 0.5*p) % p) / p - 0.5

def pipeline_features(lcr, period, t0, dur_days):
    # zero-centered flux guard: some products deliver flux with median ~0 —
    # normalize() would divide by ~0 and every feature becomes garbage. Skip the star.
    f0 = np.asarray(lcr.flux.value, float)
    m0 = np.nanmedian(f0)
    if (not np.isfinite(m0)) or m0 <= 0 or m0 < np.nanstd(f0):
        raise ValueError("zero-centered flux (unusable product)")
    lc = lcr.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)
    t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float)
    g = np.isfinite(t) & np.isfinite(f); t, f = t[g], f[g]
    durf = dur_days/period if period > 0 else 0.05
    ph = _phase(t, period, t0)
    intr = np.abs(ph) < 0.5*durf; oot = np.abs(ph) > 1.5*durf
    sigma = np.std(f[oot]) if oot.sum() > 10 else np.std(f)
    depth = (1 - np.median(f[intr])) if intr.sum() > 3 else 0.0
    snr = depth / (sigma/np.sqrt(max(intr.sum(),1)) + 1e-12)
    ep = np.round((t - t0)/period) if period > 0 else np.zeros_like(t)
    odd = intr & (ep % 2 == 1); even = intr & (ep % 2 == 0)
    d_o = (1-np.median(f[odd])) if odd.sum() > 3 else depth
    d_e = (1-np.median(f[even])) if even.sum() > 3 else depth
    oe = abs(d_o - d_e)/(depth + 1e-9)
    sec = (np.abs(ph-0.5) < 0.5*durf) | (np.abs(ph+0.5) < 0.5*durf)
    sd = (1-np.median(f[sec])) if sec.sum() > 3 else 0.0
    sr = max(sd, 0.0)/(depth + 1e-9)
    core = np.abs(ph) < 0.25*durf
    wing = (np.abs(ph) > 0.35*durf) & (np.abs(ph) < 0.5*durf)
    d_c = (1-np.median(f[core])) if core.sum() > 3 else depth
    d_w = (1-np.median(f[wing])) if wing.sum() > 3 else depth*0.5
    vs = d_w/(d_c + 1e-9)
    ntr = int(np.unique(ep[intr]).size)
    def col(*names):
        for n in names:
            if n in lcr.columns: return np.asarray(lcr[n].value, float)
        return None
    c1, c2 = col("centroid_col","mom_centr1","pos_corr1"), col("centroid_row","mom_centr2","pos_corr2")
    cen = np.nan
    if c1 is not None and c2 is not None:
        tr = np.asarray(lcr.time.value, float)
        phr = _phase(tr, period, t0)
        ri = np.abs(phr) < 0.5*durf; ro = np.abs(phr) > 1.5*durf
        gg = np.isfinite(c1) & np.isfinite(c2)
        if (ri&gg).sum() >= 3 and (ro&gg).sum() >= 3:
            sc = np.hypot(np.std(c1[ro&gg]), np.std(c2[ro&gg])) + 1e-9
            sh = np.hypot(np.median(c1[ri&gg])-np.median(c1[ro&gg]),
                          np.median(c2[ri&gg])-np.median(c2[ro&gg]))
            cen = float(sh/sc)
    # v3 anti-transit-bias features: describe the REST of the curve, not just the dip.
    oot_scatter = float(sigma)
    p2p = float(np.sqrt(np.mean(np.diff(f)**2))) if len(f) > 2 else 0.0
    return {"period":period, "depth":float(depth), "duration":dur_days,
            "duration_frac":float(durf), "snr":float(snr), "odd_even_diff":float(oe),
            "secondary_ratio":float(sr), "v_shape":float(vs), "n_transits":ntr,
            "centroid":cen, "oot_scatter":oot_scatter, "p2p_rms":p2p}''')

code('''# ---- Kepler build loop (quarter chosen to actually contain transits) ----
KOUT = f"rf4v3k_features_p{PART}.csv"           # lands in /kaggle/working -> Output panel
kdone = set()
if os.path.exists(KOUT):
    kdone = set(pd.read_csv(KOUT).uid.astype(str)); print(f"resuming — {len(kdone)} done")
rows, t0c = [], time.time()
for i, r in ksamp.iterrows():
    uid = f"K{int(r.kepid)}_{r.cls}"
    if uid in kdone: continue
    signal.alarm(TIMEOUT_S)
    try:
        sr = lk.search_lightcurve(f"KIC {int(r.kepid)}", mission="Kepler")
        if len(sr) == 0: raise ValueError("no LC")
        dur_d = float(r.koi_duration)/24.0          # KOI duration is in HOURS
        lcr, best = None, -1
        for q in range(min(4, len(sr))):
            cand = sr[q].download()
            t = np.asarray(cand.time.value, float)
            ph = ((t - float(r.koi_time0bk) + 0.5*float(r.koi_period)) % float(r.koi_period))/float(r.koi_period) - 0.5
            n_in = int((np.abs(ph) < 0.5*dur_d/float(r.koi_period)).sum())
            if n_in > best: lcr, best = cand, n_in
            if n_in >= 3: break
        feats = pipeline_features(lcr, float(r.koi_period), float(r.koi_time0bk), dur_d)
        feats.update({"uid": uid, "tic": int(r.kepid), "star": f"K{int(r.kepid)}",
                      "mission": "Kepler", "cls": r.cls})
        rows.append(feats)
    except (Exception, _TO) as e:
        print(f"  skip {uid}: {type(e).__name__}")
    finally:
        signal.alarm(0)
    if rows and (len(rows) % 25 == 0 or i == len(ksamp)-1):
        pd.DataFrame(rows).to_csv(KOUT, mode="a", header=not os.path.exists(KOUT), index=False); rows = []
        print(f"{i+1}/{len(ksamp)} · {(time.time()-t0c)/60:.0f} min · saved -> {KOUT}")
if rows: pd.DataFrame(rows).to_csv(KOUT, mode="a", header=not os.path.exists(KOUT), index=False)
built = pd.read_csv(KOUT)
print("DONE:", len(built), built.cls.value_counts().to_dict())
print(f"Download {KOUT} from the Output panel on the right (Save Version to keep it).")''')

md('''## Merge + train (run when ALL FOUR CSVs exist)
Upload the other three CSVs (`rf4v3_features_p1/p2.csv` from the Colab TESS
accounts + the other Kaggle part's `rf4v3k_features_p*.csv`) via
**+ Add Input → Upload**, then run this cell. It finds every `rf4v3*` CSV across
your inputs and this session's working dir. Grouped by star, per-mission
accuracy, and the zero-planet-loss triage threshold.''')

code('''import glob, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

FEATURES = ["period","depth","duration","duration_frac","snr",
            "odd_even_diff","secondary_ratio","v_shape","n_transits","centroid",
            "oot_scatter","p2p_rms"]                       # v3: 12 features
CLASSES  = ["transit","eclipsing_binary","blend","noise"]

fs = sorted(glob.glob("rf4v3*_features_p*.csv"))
for root, _, fls in os.walk("/kaggle/input"):
    for f in fls:
        if f.startswith("rf4v3") and f.endswith(".csv"):
            fs.append(os.path.join(root, f))
fs = sorted(set(fs))
print("using:", fs)
if len(fs) < 4:
    print(f"WARNING: only {len(fs)} of 4 CSVs found — training on what's here (partial model).")
d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset="uid").reset_index(drop=True)
# sanity filter: rows built from zero-centered flux (before the guard existed) have
# physically impossible values — drop them rather than train on poison
n0 = len(d)
d = d[(d.depth.abs() < 0.5) & (d.oot_scatter < 0.2) & (d.p2p_rms < 0.2)].reset_index(drop=True)
if n0 - len(d): print(f"sanity filter: dropped {n0-len(d)} poisoned rows (zero-centered flux artifacts)")
print("total:", len(d), d.cls.value_counts().to_dict())
print("by mission:", d.mission.value_counts().to_dict())
X = np.nan_to_num(d[FEATURES].to_numpy(float))
y = d.cls.to_numpy(); groups = d.star.to_numpy()

accs, clears, thrs = [], [], []
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for k, (tr, te) in enumerate(gss.split(X, y, groups)):
    assert len(set(groups[tr]) & set(groups[te])) == 0
    m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                               random_state=k, n_jobs=-1).fit(X[tr], y[tr])
    accs.append(accuracy_score(y[te], m.predict(X[te])))
    pt = m.predict_proba(X[te])[:, list(m.classes_).index("transit")]
    is_pl = (y[te] == "transit")
    if is_pl.sum():
        thr = pt[is_pl].min()
        thrs.append(thr); clears.append((pt < thr).mean())
print(f"grouped 5-split accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
print(f"TRIAGE (zero-planet-loss): threshold ~{np.mean(thrs):.3f} -> auto-clears "
      f"{100*np.mean(clears):.0f}% +/- {100*np.std(clears):.0f}% of curves with 0 planets discarded")

tr, te = next(gss.split(X, y, groups))
m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                           random_state=42, n_jobs=-1).fit(X[tr], y[tr])
print(classification_report(y[te], m.predict(X[te])))
print("confusion (rows=true):"); print(pd.DataFrame(confusion_matrix(y[te], m.predict(X[te]), labels=CLASSES), index=CLASSES, columns=CLASSES))
pl = (y[te] == "transit").astype(int)
auc = roc_auc_score(pl, m.predict_proba(X[te])[:, list(m.classes_).index("transit")])
print(f"binary planet-vs-rest AUC (grouped test): {auc:.3f}")
for msn in d.mission.unique():
    mask = (d.iloc[te].mission == msn).to_numpy()
    if mask.sum() > 20:
        print(f"  {msn}-only test accuracy: {accuracy_score(y[te][mask], m.predict(X[te][mask])):.3f} (n={mask.sum()})")
imp = sorted(zip(FEATURES, m.feature_importances_), key=lambda x: -x[1])
print("top features:", [(f, round(v,3)) for f, v in imp[:6]])

final = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                               random_state=42, n_jobs=-1).fit(X, y)
joblib.dump({"model": final, "features": FEATURES}, "model_rf4v3.joblib")
print("saved model_rf4v3.joblib -> download from the Output panel")
print("deploy: replace tara/backend/app/models/tabular/model.joblib (back up the old one first)")
print("SUCCESS CRITERIA vs v2: transit precision 0.63 -> 0.75+, blend recall 0.48 -> 0.65+")''')

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": CELLS}
out = __file__.replace("make_kaggle_kepler_build.py", "tara_kaggle_kepler_build.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
