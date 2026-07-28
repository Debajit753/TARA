"""Generates tara_colab_period_test.ipynb — the TOI-1022 period-recovery experiment.

Question it answers in ~5 min on Colab (CPU): does a PROPER full-range BLS/TLS
search recover TOI-1022's true period (3.097 d) where TARA's fast two-stage
search locked a 4.93 d alias? Downloads the real light curve, runs a full BLS +
full TLS, folds on all three periods side-by-side, and prints a plain verdict.

Ground truth (SPOC Data Validation Summary): P = 3.097 d, T0 = 1547.448 BTJD,
depth ~330 ppm, duration ~5.24 hr. TARA's fast search reported 4.93 d.
"""
import json

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — period-recovery test: can a proper search find TOI-1022?

**The question.** TARA's *fast* two-stage search reported **4.93 d** for TOI-1022
(TIC 47384844). NASA's SPOC pipeline says the truth is **3.097 d** (a real
sub-Neptune candidate). Was TARA wrong because the star is genuinely too hard —
or just because the *fast* search cut a corner?

This notebook runs a **proper full-range BLS and TLS** on the same light curve and
folds on all three periods. If the full search finds 3.097 d, the fix is simply to
widen TARA's search. If it also misses, the star is genuinely hard.

Just **Runtime → Run all** (CPU is fine, ~3–6 min). No uploads.''')

code('''!pip install -q lightkurve transitleastsquares''')

code('''# ---- download + clean the SAME way TARA does ----
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt, lightkurve as lk

TIC   = "TIC 47384844"     # = TOI-1022
P_TARA = 4.9300            # what TARA's fast search reported
P_TRUE = 3.097             # SPOC Data Validation Summary
T0_TRUE = 1547.448         # BTJD (BJD - 2457000), from the DVS

lcf = lk.search_lightcurve(TIC, mission="TESS", author="SPOC")[0].download()
lc  = lcf.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)
t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float)
print(f"{TIC}: {len(t)} points, sector baseline {t.max()-t.min():.0f} d")
plt.figure(figsize=(11,2.4)); plt.plot(t, f, ".", ms=2, alpha=.5)
plt.title(f"{TIC} — detrended light curve"); plt.xlabel("time (BTJD)"); plt.ylabel("flux"); plt.show()''')

md('''## 1. Full BLS — the whole period grid, not just the tallest peak
TARA's fast search takes only the single highest BLS peak and refines around it.
Here we look at the **whole periodogram** and check the power exactly at 3.097 d.''')

code('''bls = lc.to_periodogram(method="bls", minimum_period=0.5, maximum_period=8.0, frequency_factor=500)
P = bls.period.value; PW = bls.power.value
p_bls = float(bls.period_at_max_power.value)

def power_at(period):
    return float(PW[np.argmin(np.abs(P - period))])

print(f"BLS single tallest peak : {p_bls:.4f} d   (power {PW.max():.1f})")
print(f"BLS power at 3.097 d TRUE: {power_at(P_TRUE):.1f}")
print(f"BLS power at 4.93 d TARA : {power_at(P_TARA):.1f}")

plt.figure(figsize=(11,3))
plt.plot(P, PW, lw=1, color="#4f46e5")
for p,c,lab in [(P_TRUE,"#0d9488","true 3.097 d"),(P_TARA,"#b45309","TARA 4.93 d")]:
    plt.axvline(p, color=c, ls="--", lw=1.5, label=lab)
plt.title("Full BLS periodogram"); plt.xlabel("period (days)"); plt.ylabel("BLS power")
plt.legend(); plt.show()''')

md('''## 2. Full TLS — realistic transit shape, whole range
TLS fits a real limb-darkened transit (more sensitive than BLS's box). This is the
"thorough" search — slower, but it evaluates every period properly.''')

code('''from transitleastsquares import transitleastsquares
tt = np.ascontiguousarray(t); ff = np.ascontiguousarray(f)
m = np.isfinite(tt) & np.isfinite(ff)
print("running full TLS (≈1–3 min on Colab)…")
res = transitleastsquares(tt[m], ff[m]).power(
    period_min=0.5, period_max=8.0, oversampling_factor=3,
    duration_grid_step=1.05, use_threads=2, show_progress_bar=True)
print(f"\\nTLS best period : {res.period:.4f} d")
print(f"TLS SDE         : {res.SDE:.1f}  (>~7 = a real detection)")
print(f"TLS depth       : {(1-res.depth)*1e6:.0f} ppm   (SPOC: ~330 ppm)")

plt.figure(figsize=(11,3))
plt.plot(res.periods, res.power, lw=1, color="#4f46e5")
for p,c,lab in [(P_TRUE,"#0d9488","true 3.097 d"),(P_TARA,"#b45309","TARA 4.93 d")]:
    plt.axvline(p, color=c, ls="--", lw=1.5, label=lab)
plt.title("Full TLS periodogram (SDE)"); plt.xlabel("period (days)"); plt.ylabel("SDE")
plt.legend(); plt.show()''')

md('''## 3. Fold on all three periods — the eyeball test
A folded curve on the RIGHT period shows one clean dip; on a wrong period the
transit smears into noise.''')

code('''def fold_plot(ax, period, t0, title, color):
    ph = ((t - t0 + 0.5*period) % period)/period - 0.5
    o = np.argsort(ph)
    ax.plot(ph[o], f[o], ".", ms=2, alpha=.35, color="#9aa1af")
    # binned medians
    nb=50; edges=np.linspace(-0.5,0.5,nb+1); idx=np.clip(np.digitize(ph,edges)-1,0,nb-1)
    bx=[]; by=[]
    for b in range(nb):
        v=f[idx==b]
        if len(v)>=3: bx.append(-0.5+(b+0.5)/nb); by.append(np.median(v))
    ax.plot(bx,by,"o",ms=3,color=color)
    ax.set_title(title); ax.set_xlim(-0.5,0.5); ax.set_xlabel("phase")

fig,axs=plt.subplots(1,3,figsize=(13,3.2),sharey=True)
fold_plot(axs[0], P_TARA, res.T0, f"TARA fast: {P_TARA} d", "#b45309")
fold_plot(axs[1], res.period, res.T0, f"Full TLS: {res.period:.4f} d", "#4f46e5")
fold_plot(axs[2], P_TRUE, T0_TRUE, f"SPOC truth: {P_TRUE} d", "#0d9488")
plt.tight_layout(); plt.show()''')

md('''## 4. Harmonic check + the verdict
Also test the tallest BLS peak's half and double (P, P/2, 2P) — the classic
anti-alias trick — then print the plain-language conclusion.''')

code('''cands = sorted({round(x,3) for x in [p_bls, p_bls/2, p_bls*2, res.period, P_TARA]} )
print("period    | matches TRUE 3.097 d?")
for c in cands:
    print(f"  {c:6.3f} d | {'✅ YES' if abs(c-P_TRUE)<0.06 else '—'}")

got = abs(res.period - P_TRUE) < 0.06 or abs(p_bls - P_TRUE) < 0.06
print("\\n" + "="*60)
if got:
    print("VERDICT: a PROPER full search DOES recover 3.097 d.")
    print("-> TARA's miss was the FAST two-stage shortcut, not the data.")
    print("   Fix: widen TARA's search grid / add the multi-peak check.")
else:
    print("VERDICT: even a full BLS/TLS does NOT cleanly recover 3.097 d here.")
    print(f"   (full TLS SDE {res.SDE:.1f}) -> genuinely hard shallow candidate;")
    print("   TARA's 'uncertain' + human review queue is the right call.")
print("="*60)''')

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": CELLS}
out = __file__.replace("make_colab_period_test.py", "tara_colab_period_test.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
