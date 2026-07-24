# TARA — Progress checkpoint
**Updated:** 2026-07-05 · *local project checkpoint*

> Fresh chat? Say **"continue the TARA exoplanet build"** — memory holds full state.
> This is the short status view; see `architecture.md` for the full technical design.

## Status: build complete

### What serves the live API today
- **4-class RandomForest** (transit / eclipsing binary / blend / noise) — `models/tabular/model.joblib`,
  73.6% grouped accuracy, planet-vs-rest AUC 0.906, trained on 2,055 ExoMiner-labeled stars
  via the live pipeline's own 10 features. No-detection guard at SNR ≥ 7.0 (field standard).
- **Cross-mission CNN ensemble** (Kepler+TESS mixed, 5 seeds) — AUC **0.951** held-out
  (Kepler 0.932 / TESS 0.966), the headline model, second opinion on every star.
- **±1σ uncertainties** from trapezoid fitting; batman+emcee MCMC validated on WASP-18.

### UI (the website = `frontend/`)
- **`frontend/dash-workspace.html`** — live review workspace: queue with folders
  (Demo/Searched/Uploads), 5 tabs incl. real pipeline timings, dark mode, CSV/FITS import
  (FITS upload via `POST /analyze-file`), hover tooltips.
- `frontend/about.html` — what/how/model story with the honest numbers.
- Old demo/ pages (live.html, index.html, concept dashboards) deleted 2026-07-05 — superseded.
- The old React app was deleted; the vanilla HTML dashboard now lives in `frontend/`.

### Next levers (when time permits)
1. Run the **Kepler DR25 blend build** (`notebooks/tara_colab_rf4_build.ipynb`, optional
   section) — 2,162 Kepler blends vs 258 now → fixes the weakest class.
2. **Deploy**: GitHub (storage) + Hugging Face Space (engine, free CPU) — ~7 MB footprint.
3. Deck/report: results + honest AUC hierarchy are all measured and written up.

### Known limitations (honest)
- Photometric ceiling: some FPs are indistinguishable without follow-up.
- Savgol detrend can absorb deep eclipses → search can lock a half-period (seen live).
- Cross-catalog domain shift ~0.72 (fix: retrain on a curated in-distribution dataset).

## Run
```bash
cd tara && .venv/bin/uvicorn app.main:app --app-dir backend --port 8000   # API (~1-2 min warm-up)
.venv/bin/python backend/precompute_cache.py                              # fill demo cache
npx http-server "frontend" -p 5196                                       # UI → /dash-workspace.html
```
Demo playbook: boot + precompute 30 min before stage → cached stars instant, live star ~15–90 s.
