"""Generates tara_colab_calibrate.ipynb — TEMPERATURE CALIBRATION for the mixed
cross-mission CNN ensemble (fixes the documented overconfidence: ~0.99 for
anything planet-shaped).

Perfected for a real Colab run:
- mounts Google Drive and AUTO-FINDS every needed file (no 80 MB browser uploads);
  falls back to upload prompts only for what's missing
- auto-detects the scalar-feature count from the shipped scaler (NS = len(mu)) —
  works with any model variant, 11 features or otherwise
- uses however many cnn_mixed_seed*.pt files exist (warns if fewer than 5)
- reconstructs mixed_train's EXACT grouped split (same files, K-then-T order,
  GroupShuffleSplit random_state=42) and VERIFIES against test_groups_mixed.npy
  when available
- clear, named errors for every missing/mismatched input

Method (Guo et al. 2017, the field standard): fit ONE scalar T on the grouped
held-out set; serving divides the ensemble's mean logit by T. AUC 0.951 is
mathematically unchanged — only the probability becomes honest.

CPU runtime, ~5-10 min. Output: calibration.json -> drop into
tara/backend/app/models/mixed/ and restart the backend (serving auto-detects it).
"""
import json

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — calibrate the mixed CNN ensemble (temperature scaling)

Fixes the CNN's overconfidence WITHOUT retraining: fits one number **T** on the
grouped held-out set; serving divides the mean logit by T. AUC unchanged —
only the probabilities become honest.

**Files it needs** (the next cell mounts your Drive and finds them automatically):
| File | Usually lives |
|---|---|
| `kepler_views_merged.npz` | Drive (from merge-all) / project `data/processed` |
| `tce_views_merged.npz` | Drive (from the tce merge) |
| `cnn_mixed_seed*.pt` (any number, ideally 5) | Drive / project `models/mixed` |
| `scalar_norm_mixed.npz` | same place as the seeds |
| `test_groups_mixed.npy` *(optional — verifies the split)* | Drive |

**CPU runtime is fine. Run all.** Anything it can't find, it asks you to upload.
Output: **calibration.json** → put in `tara/backend/app/models/mixed/`, restart
the backend — the dashboard's CNN row switches to "calibrated" automatically.''')

code('''# ---- find every input: Drive first, upload fallback ----
import os, glob, shutil
USE_DRIVE = True          # set False to skip Drive and upload everything manually

NEEDED   = ["kepler_views_merged.npz", "tce_views_merged.npz", "scalar_norm_mixed.npz"]
OPTIONAL = ["test_groups_mixed.npy"]

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", type(e).__name__, e)

def hunt():
    found, seeds = {}, {}
    for root in ["/content", "/content/drive/MyDrive"]:
        if not os.path.isdir(root): continue
        for dirpath, dirs, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 3:   # don't crawl the whole Drive
                dirs[:] = []; continue
            for f in files:
                if f in NEEDED + OPTIONAL and f not in found:
                    found[f] = os.path.join(dirpath, f)
                if f.startswith("cnn_mixed_seed") and f.endswith(".pt") and f not in seeds:
                    seeds[f] = os.path.join(dirpath, f)
    return found, seeds

found, seeds = hunt()
for f, p in {**found, **seeds}.items():
    if not os.path.exists(f):
        print(f"  found {f}  <-  {p}")
        shutil.copy(p, f)

missing = [f for f in NEEDED if not os.path.exists(f)]
if missing or not glob.glob("cnn_mixed_seed*.pt"):
    from google.colab import files
    print("\\nNot found automatically — please upload:", missing,
          "" if glob.glob("cnn_mixed_seed*.pt") else "+ the cnn_mixed_seed*.pt files")
    files.upload()

missing = [f for f in NEEDED if not os.path.exists(f)]
assert not missing, f"still missing: {missing} — upload them and re-run this cell"
SEEDS = sorted(glob.glob("cnn_mixed_seed*.pt"))
assert SEEDS, "no cnn_mixed_seed*.pt files found — upload the seed models and re-run"
if len(SEEDS) < 5: print(f"NOTE: only {len(SEEDS)} of 5 seeds found — calibrating what's here.")
print("\\nready:", NEEDED, "| seeds:", SEEDS,
      "| split verification file:", os.path.exists("test_groups_mixed.npy"))''')

code('''# ---- load the SAME data the mixed model was trained on, in the SAME order ----
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

kp = np.load("kepler_views_merged.npz", allow_pickle=True)
tv = np.load("tce_views_merged.npz",   allow_pickle=True)
for name, z in [("kepler_views_merged", kp), ("tce_views_merged", tv)]:
    miss = [k for k in ("G","L","Y","SC","K") if k not in z.files]
    assert not miss, f"{name}.npz is missing arrays {miss} — wrong file?"

G = np.concatenate([kp["G"], tv["G"]]).astype("float32")
L = np.concatenate([kp["L"], tv["L"]]).astype("float32")
Y = np.concatenate([kp["Y"], tv["Y"]]).astype("float32")
S = np.concatenate([kp["SC"], tv["SC"]]).astype("float64")
groups = np.array([f"K{int(k)}" for k in kp["K"]] + [f"T{int(k)}" for k in tv["K"]])
print(f"rows: {len(Y)} (Kepler {len(kp['Y'])} + TESS {len(tv['Y'])}) | scalar features per row: {S.shape[1]}")

idx = np.arange(len(Y))
itr, ite = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(idx, Y, groups=groups))
assert len(set(groups[itr]) & set(groups[ite])) == 0, "star overlap must be 0"
print(f"reconstructed grouped split: {len(itr)} train / {len(ite)} held-out")
if os.path.exists("test_groups_mixed.npy"):
    saved = set(np.load("test_groups_mixed.npy", allow_pickle=True).tolist())
    now = set(np.unique(groups[ite]).tolist())
    if saved == now:
        print("VERIFIED: split matches mixed_train's saved held-out groups exactly.")
    else:
        print(f"WARNING: split differs from saved test groups (overlap "
              f"{len(saved & now)}/{len(saved)}) — calibration still valid but "
              "fitted on a slightly different held-out set.")''')

code('''# ---- model: scalar count auto-detected from the shipped scaler ----
import torch, torch.nn as nn

nz = np.load("scalar_norm_mixed.npz")
mu, sd, med = nz["mu"], nz["sd"], nz["med"]
NS = len(mu)                                   # take ALL the features the model has
assert S.shape[1] >= NS, f"views have {S.shape[1]} scalars but scaler expects {NS}"
if S.shape[1] > NS:
    print(f"NOTE: views carry {S.shape[1]} scalars, model uses the first {NS} — trimming.")
    S = S[:, :NS]
print(f"scalar features: {NS} (auto-detected from scalar_norm_mixed.npz)")

Sn = np.where(np.isnan(S), med, S)
Sn = np.where(sd < 1e-6, 0.0, (Sn - mu) / np.where(sd < 1e-6, 1.0, sd))
Sn = np.clip(np.nan_to_num(Sn), -10, 10).astype("float32")

def block(ci, co, k=5, pool=4):
    return [nn.Conv1d(ci, co, k, padding=k//2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(pool)]
class Net(nn.Module):
    def __init__(self, ns=NS, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g = nn.Sequential(*block(1,16), *block(16,32), *block(32,64), nn.AdaptiveMaxPool1d(8), nn.Flatten())
        self.l = nn.Sequential(*block(1,16,pool=2), *block(16,32,pool=2), nn.AdaptiveMaxPool1d(8), nn.Flatten())
        self.s = nn.Sequential(nn.Linear(ns, sh), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64*8+32*8+sh, hd), nn.ReLU(), nn.Dropout(drop), nn.Linear(hd, 1))
    def forward(self, g, l, s):
        return self.head(torch.cat([self.g(g), self.l(l), self.s(s)], 1)).squeeze(1)

nets = []
for p in SEEDS:
    n = Net()
    try:
        n.load_state_dict(torch.load(p, map_location="cpu"))
    except RuntimeError as e:
        raise SystemExit(f"{p} does not match the mixed architecture ({NS} scalars) — "
                         f"wrong model family?\\n{e}")
    n.eval(); nets.append(n)
print(f"loaded {len(nets)} seed models, all architecture-verified")''')

code('''# ---- held-out logits -> fit T -> honest before/after report -> calibration.json ----
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
import json

def logits(net, sel, bs=512):
    out = []
    with torch.no_grad():
        for a in range(0, len(sel), bs):
            b = sel[a:a+bs]
            out.append(net(torch.tensor(G[b]).unsqueeze(1), torch.tensor(L[b]).unsqueeze(1),
                           torch.tensor(Sn[b])).numpy())
    return np.concatenate(out)

Z = np.stack([logits(n, ite) for n in nets])       # (n_seeds, n_test) raw logits
zbar = Z.mean(0)
probs_uncal = 1/(1+np.exp(-Z))
spread = probs_uncal.std(0)
yte = Y[ite]
print(f"held-out n={len(yte)} | mean seed spread {spread.mean():.3f}")

def nll(T):
    p = np.clip(1/(1+np.exp(-zbar/T)), 1e-7, 1-1e-7)
    return -np.mean(yte*np.log(p) + (1-yte)*np.log(1-p))
T = float(minimize_scalar(nll, bounds=(0.25, 25.0), method="bounded").x)

def ece(p, y, bins=10):
    e = 0.0
    for b in range(bins):
        m = (p >= b/bins) & (p < (b+1)/bins)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
p0 = 1/(1+np.exp(-zbar)); p1 = 1/(1+np.exp(-zbar/T))
print(f"\\nTEMPERATURE T = {T:.2f}   (T > 1 confirms the raw model was overconfident)")
print(f"ECE (calibration error): {ece(p0,yte):.3f} -> {ece(p1,yte):.3f}")
print(f"Brier score            : {np.mean((p0-yte)**2):.3f} -> {np.mean((p1-yte)**2):.3f}")
print(f"AUC unchanged          : {roc_auc_score(yte,p0):.4f} == {roc_auc_score(yte,p1):.4f}")
print("\\nreliability (predicted vs actual planet fraction, calibrated):")
for b in range(10):
    m = (p1 >= b/10) & (p1 < (b+1)/10)
    if m.sum() > 5: print(f"  {b/10:.1f}-{(b+1)/10:.1f}: predicted {p1[m].mean():.2f} | actual {yte[m].mean():.2f} (n={m.sum()})")

json.dump({"temperature": round(T, 4), "n_scalars": int(NS), "n_seeds": len(nets),
           "fitted_on": "grouped held-out, random_state=42", "n_test": int(len(yte)),
           "ece_before": round(float(ece(p0,yte)),4), "ece_after": round(float(ece(p1,yte)),4)},
          open("calibration.json","w"), indent=1)
from google.colab import files
files.download("calibration.json")
print("\\nDEPLOY: put calibration.json in tara/backend/app/models/mixed/ and restart the backend.")''')

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": CELLS}
out = __file__.replace("make_colab_calibrate.py", "tara_colab_calibrate.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
