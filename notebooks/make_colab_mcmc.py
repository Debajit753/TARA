"""Generates tara_colab_mcmc.ipynb — rigorous transit fit with batman + emcee.

The live API uses a fast trapezoid fit for error bars (cold, milliseconds). THIS
notebook does the heavy Bayesian version for a hero star: MCMC over a physical
batman transit model -> full posteriors + corner plot + credible-band fit, with
proper (possibly asymmetric) uncertainties on period, Rp/Rs, depth, a/Rs, inc and
derived transit duration. CPU-bound (emcee) -> run on Colab, not a fanless laptop.
A plain CPU runtime is fine."""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — rigorous transit fit (batman + emcee MCMC)

Bayesian parameter estimation for one hero star: fits a physical limb-darkened
transit model and returns **full posteriors** (median ± 1σ, and a corner plot),
not just point values. This is the "estimation by light-curve fitting" deliverable
done properly.

**CPU runtime is fine** (emcee is CPU-bound; no GPU needed). ~2–4 min.
Change `TARGET` / `MISSION` to fit a different star.''')

code('''!pip install -q lightkurve batman-package emcee corner''')

code('''MISSION = "TESS"            # "TESS" or "Kepler"
TARGET  = "TIC 100100827"    # WASP-18 (deep, clean hot Jupiter) — change as needed
NWALKERS, NSTEPS, DISCARD = 32, 3000, 1000
import numpy as np, lightkurve as lk

sr = lk.search_lightcurve(TARGET, mission=MISSION)
lc = sr[0].download().remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)
t = np.ascontiguousarray(lc.time.value, float)
f = np.ascontiguousarray(lc.flux.value, float)
print(f"{TARGET}: {len(t)} points over {t.max()-t.min():.1f} days")''')

code('''# find period/t0/duration with TLS to seed the fit
from transitleastsquares import transitleastsquares
res = transitleastsquares(t, f).power(period_min=0.5, period_max=10,
        oversampling_factor=3, duration_grid_step=1.1, show_progress_bar=False)
P0, T0, DUR = float(res.period), float(res.T0), float(res.duration)
DEPTH0 = float(abs(1 - res.depth))
print(f"seed: P={P0:.5f} d, T0={T0:.4f}, duration={DUR*24:.2f} h, depth={DEPTH0*1e6:.0f} ppm, SDE={res.SDE:.1f}")

# keep only points near transit (speeds up MCMC a lot)
ph = ((t - T0 + 0.5*P0) % P0)/P0 - 0.5
win = min(0.5, max(3*DUR/P0, 0.03))
m = np.abs(ph) < win
tt, ff = t[m], f[m]
sig = np.std(ff[np.abs(ph[m]) > 0.5*win]) or np.std(ff)
print(f"fitting {len(tt)} near-transit points, noise sigma {sig*1e6:.0f} ppm")''')

code('''import batman
# a/Rs seed from duration: T14 ~ P/(pi*a)  ->  a ~ P/(pi*T14)
A0 = max(P0 / (np.pi * DUR), 2.0)
RP0 = np.sqrt(DEPTH0)

pm = batman.TransitParams()
pm.t0, pm.per, pm.rp, pm.a = T0, P0, RP0, A0
pm.inc, pm.ecc, pm.w = 89.0, 0.0, 90.0
pm.u, pm.limb_dark = [0.4, 0.3], "quadratic"
bat = batman.TransitModel(pm, tt)          # bound to tt once (fast to re-evaluate)

def model(theta):
    per, t0, rp, a, inc = theta
    pm.per, pm.t0, pm.rp, pm.a, pm.inc = per, t0, rp, a, min(inc, 90.0)
    return bat.light_curve(pm)

def log_prob(theta):
    per, t0, rp, a, inc = theta
    if not (P0*0.98 < per < P0*1.02): return -np.inf
    if not (1e-3 < rp < 0.5):         return -np.inf
    if not (1.5 < a < 60):            return -np.inf
    if not (75 < inc <= 90):          return -np.inf
    if abs(t0 - T0) > DUR:            return -np.inf
    r = ff - model(theta)
    return -0.5 * np.sum((r/sig)**2)

import emcee
p0 = np.array([P0, T0, RP0, A0, 89.0])
scale = np.array([1e-4*P0, 1e-3, 0.05*RP0, 0.1*A0, 0.5])
pos = p0 + scale * np.random.randn(NWALKERS, 5)
pos[:,4] = np.clip(pos[:,4], 76, 90)
sampler = emcee.EnsembleSampler(NWALKERS, 5, log_prob)
sampler.run_mcmc(pos, NSTEPS, progress=True)
flat = sampler.get_chain(discard=DISCARD, thin=10, flat=True)
print("posterior samples:", flat.shape[0])''')

code('''labels = ["period (d)", "t0", "Rp/Rs", "a/Rs", "inc (deg)"]
pct = np.percentile(flat, [16, 50, 84], axis=0)
print("=== posterior (median +/- 1sigma) ===")
for i, lab in enumerate(labels):
    lo, mid, hi = pct[0,i], pct[1,i], pct[2,i]
    print(f"  {lab:12}: {mid:.5g}  (+{hi-mid:.2g} / -{mid-lo:.2g})")

# derived: depth (ppm) and transit duration T14 (hours) with uncertainties from samples
per_s, t0_s, rp_s, a_s, inc_s = flat.T
depth_s = rp_s**2 * 1e6
b_s = a_s*np.cos(np.radians(inc_s))
arg = np.clip(((1+rp_s)**2 - b_s**2), 0, None)
T14_s = (per_s/np.pi)*np.arcsin(np.clip(np.sqrt(arg)/(a_s*np.sin(np.radians(inc_s))),-1,1))*24
def q(x): p=np.percentile(x,[16,50,84]); return p[1],p[2]-p[1],p[1]-p[0]
for lab, s, u in [("depth (ppm)", depth_s, ""), ("duration (h)", T14_s, "")]:
    mid, ep, em = q(s); print(f"  {lab:12}: {mid:.4g}  (+{ep:.2g} / -{em:.2g})")''')

code('''import corner, matplotlib.pyplot as plt
fig = corner.corner(flat, labels=labels, quantiles=[0.16,0.5,0.84],
                    show_titles=True, title_fmt=".4g", title_kwargs={"fontsize":9})
fig.suptitle(f"{TARGET} — transit posterior", y=1.02)
plt.show()''')

code('''# folded data + MCMC median model + credible band
import matplotlib.pyplot as plt
phf = ((tt - pct[1,1] + 0.5*pct[1,0]) % pct[1,0])/pct[1,0] - 0.5
order = np.argsort(phf)
draws = flat[np.random.randint(len(flat), size=120)]
band = np.array([model(th) for th in draws])
med_model = model(pct[1])
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(phf, ff, ".", ms=2, alpha=.35, color="#4f46e5", label="TESS flux")
lo, hi = np.percentile(band, [16,84], axis=0)
ax.fill_between(phf[order], lo[order], hi[order], color="#d97706", alpha=.3, label="68% model band")
ax.plot(phf[order], med_model[order], color="#b45309", lw=2, label="MCMC median fit")
ax.set_xlabel("orbital phase"); ax.set_ylabel("normalized flux")
ax.set_title(f"{TARGET} — phase-folded transit + Bayesian fit"); ax.legend()
ax.set_xlim(-win, win); plt.show()''')

md('''## Using this
- Quote parameters as **median +1σ / −1σ** straight from the posterior — that is the rigorous "estimation by light-curve fitting" the brief asks for.
- The **corner plot** shows parameter correlations (e.g. a/Rs ↔ inclination degeneracy) — good evidence of a real, physical fit rather than a black box.
- The **credible band** on the folded curve is the figure to put next to the classifier verdict in the report/demo.
- The live API keeps the fast trapezoid error bars for interactivity; this notebook is the deep version for the hero star(s).''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tara_colab_mcmc.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
