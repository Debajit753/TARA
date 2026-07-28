# Models

Organized by model family. Serving auto-selects the active CNN ensemble (`mixed/`) via `app/cnn_infer.py`; the Random Forest primary label classifier loads from `tabular/model.joblib` (`app/main.py`).

| Folder | Contents | Role |
|---|---|---|
| **mixed/** | `cnn_mixed_seed1-5.pt` + `scalar_norm_mixed.npz` + `calibration.json` | **ACTIVE — serves /analyze.** Cross-mission (Kepler+TESS 2-view CNN), AUC 0.951 |
| **tabular/** | `model.joblib` | **ACTIVE — serves Random Forest primary label classifier** |

**Note:** If you retrain models via the scripts in `backend/train/` or the notebooks in `notebooks/`, move new `cnn_*_seed*.pt` weights into `mixed/` or update `tabular/model.joblib` accordingly.

