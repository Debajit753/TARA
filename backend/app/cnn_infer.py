"""Multimodal CNN ensemble inference for serving.

Loads the best available model set (see Ensemble.PROFILES — the cross-mission
Kepler+TESS mixed ensemble first, AUC 0.951). Predicts P(planet) from the two
views + an 11-value scalar vector, averaging the seed models and thresholding
at 0.45 (the tuned operating point).

Serving honesty note: the model was trained on Kepler KOI catalog scalars. For a
live TESS star we can compute 8 of the 11 for real (period, duration, depth, SNR,
planet-radius-from-depth, and stellar Teff/logg/radius from the FITS header); the
remaining 3 (impact, equilibrium temp, insolation) have no live equivalent and
are median-filled. assemble_scalars() reports how many were real so the caller
can label the verdict truthfully.
"""
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn

# training scalar order (Kepler koi_* columns)
SCCOLS = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr",
          "koi_impact", "koi_steff", "koi_slogg", "koi_srad", "koi_teq", "koi_insol"]
THRESHOLD = 0.45


def _block(ci, co, k=5, pool=4):
    return [nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(pool)]


class Net(nn.Module):
    def __init__(self, ns=11, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g = nn.Sequential(*_block(1, 16), *_block(16, 32), *_block(32, 64),
                               nn.AdaptiveMaxPool1d(8), nn.Flatten())
        self.l = nn.Sequential(*_block(1, 16, pool=2), *_block(16, 32, pool=2),
                               nn.AdaptiveMaxPool1d(8), nn.Flatten())
        self.ns = ns
        if ns:
            self.s = nn.Sequential(nn.Linear(ns, sh), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * 8 + 32 * 8 + (sh if ns else 0), hd),
                                  nn.ReLU(), nn.Dropout(drop), nn.Linear(hd, 1))

    def forward(self, g, l, s=None):
        z = [self.g(g), self.l(l)]
        if self.ns:
            z.append(self.s(s))
        return self.head(torch.cat(z, 1)).squeeze(1)


def assemble_scalars(feats, meta):
    """Best-effort 11-value scalar vector from live pipeline features + LC header.
    Returns (vec, n_real). Unavailable entries are NaN (median-filled at predict)."""
    depth = max(float(feats.get("depth", 0.0) or 0.0), 0.0)
    srad = meta.get("radius")
    teff = meta.get("teff")
    logg = meta.get("logg")
    # planet radius (Earth radii) from depth & stellar radius: Rp = sqrt(depth)*Rstar*109.2
    prad = (np.sqrt(depth) * float(srad) * 109.2) if srad else np.nan
    vec = [
        float(feats.get("period", np.nan)),          # koi_period  (days)
        float(feats.get("duration", 0.0)) * 24.0,    # koi_duration (hours)
        depth * 1e6,                                  # koi_depth   (ppm)
        prad,                                         # koi_prad    (R_earth, estimated)
        float(feats.get("snr", np.nan)),             # koi_model_snr
        np.nan,                                       # koi_impact  (no live equivalent)
        float(teff) if teff else np.nan,             # koi_steff
        float(logg) if logg else np.nan,             # koi_slogg
        float(srad) if srad else np.nan,             # koi_srad
        np.nan,                                       # koi_teq     (no live equivalent)
        np.nan,                                       # koi_insol   (no live equivalent)
    ]
    n_real = int(np.sum(~np.isnan(vec)))
    return np.array(vec, dtype=float), n_real


class Ensemble:
    # priority: cross-mission mixed (AUC 0.951, the headline) -> TESS fine-tuned
    # -> Kepler merged -> Kepler 2k. Each model set is PAIRED with its scaler.
    PROFILES = [
        ("mixed/cnn_mixed_seed*.pt",              "mixed/scalar_norm_mixed.npz",          "Kepler+TESS", "cross-mission mixed ensemble"),
        ("tess_finetuned/cnn_tess_finetuned*.pt", "tess_finetuned/scalar_norm_tess.npz",  "TESS", "TESS fine-tuned ensemble"),
        ("kepler_merged/cnn_merged_seed*.pt",     "kepler_merged/scalar_norm_merged.npz", "Kepler", "Kepler merged ensemble"),
        ("kepler_2k/cnn_2k_multimodal.pt",        "kepler_2k/scalar_norm_2k.npz",         "Kepler", "Kepler 2k model"),
    ]

    def __init__(self, model_dir):
        seeds, norm_file = [], None
        self.trained_on, self.source = "none", "none"
        for pat, nf, trained, label in self.PROFILES:
            found = sorted(glob.glob(os.path.join(model_dir, pat)))
            if found and os.path.exists(os.path.join(model_dir, nf)):
                seeds, norm_file, self.trained_on, self.source = found, nf, trained, label
                break
        self.nets = []
        for s in seeds:
            net = Net(len(SCCOLS))
            net.load_state_dict(torch.load(s, map_location="cpu"))
            net.eval()
            self.nets.append(net)
        if norm_file:
            nz = np.load(os.path.join(model_dir, norm_file))
            self.mu, self.sd, self.med = nz["mu"], nz["sd"], nz["med"]
        self.n_models = len(self.nets)
        # optional temperature calibration (calibration.json next to the scaler,
        # produced by tara_colab_calibrate.ipynb) — fixes overconfidence, AUC unchanged
        self.temperature = None
        if norm_file:
            cal = os.path.join(model_dir, os.path.dirname(norm_file), "calibration.json")
            if os.path.exists(cal):
                try:
                    self.temperature = float(json.load(open(cal))["temperature"])
                except Exception:
                    self.temperature = None

    def predict(self, g, l, scalars):
        s = np.where(np.isnan(scalars), self.med, scalars)
        with np.errstate(divide="ignore", invalid="ignore"):
            sn = (s - self.mu) / self.sd
        # features that were constant during training (sd~0) carry no info -> zero them,
        # and clip so any live scalar out of the training range can't blow up the model
        sn = np.where(self.sd < 1e-6, 0.0, sn)
        sn = np.clip(np.nan_to_num(sn), -10.0, 10.0).astype("float32")
        gt = torch.tensor(g, dtype=torch.float32).view(1, 1, -1)
        lt = torch.tensor(l, dtype=torch.float32).view(1, 1, -1)
        st = torch.tensor(sn, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            zs = [float(net(gt, lt, st)[0]) for net in self.nets]      # raw logits
        ps = [1.0 / (1.0 + np.exp(-z)) for z in zs]                    # per-seed probs
        if self.temperature:
            # calibrated path: mean logit / T (rankings unchanged, probabilities honest)
            prob = float(1.0 / (1.0 + np.exp(-np.mean(zs) / self.temperature)))
            # Spread must live on the SAME scale as the probability it annotates.
            # Measuring it on raw (uncalibrated) probs made the displayed "±" far
            # too tight on exactly the confident calls, where temperature scaling
            # moves probabilities most.
            spread_src = [1.0 / (1.0 + np.exp(-z / self.temperature)) for z in zs]
        else:
            prob = float(np.mean(ps))                                  # legacy: mean of probs
            spread_src = ps
        return {
            "planet_probability": round(prob, 3),
            "seed_spread": round(float(np.std(spread_src)), 3),   # disagreement across the 5 seeds
            "calibrated": bool(self.temperature),
            "label": "planet" if prob > THRESHOLD else "not_planet",
            "threshold": THRESHOLD,
            "n_models": self.n_models,
            "source": self.source,
            "trained_on": self.trained_on,
        }
