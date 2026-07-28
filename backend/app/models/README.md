# Models

Organized by model family. Serving auto-selects the best available CNN ensemble
(top of the list) via `app/cnn_infer.py` `Ensemble.PROFILES`; the RandomForest
primary label loads from `tabular/model.joblib` (`app/main.py`).

| Folder | Contents | Role |
|---|---|---|
| **mixed/** | `cnn_mixed_seed1-5.pt` + `scalar_norm_mixed.npz` | **ACTIVE — serves /analyze.** Cross-mission (Kepler+TESS), AUC 0.951 |
| tess_finetuned/ | `cnn_tess_finetuned_seed1-5.pt` + `scalar_norm_tess.npz` | TESS-only fine-tune (fallback) |
| kepler_merged/ | `cnn_merged_seed1-5.pt` + `scalar_norm_merged.npz` | Kepler-only ensemble; the pretrain base for fine-tunes |
| kepler_2k/ | `cnn_2k_multimodal.pt` + `scalar_norm_2k.npz` | early 2k-star Kepler model (fallback) |
| tabular/ | `model.joblib` (TESS shape RF, primary label), `kepler_koi.joblib` (Kepler catalog RF, 86.9%) | non-CNN classifiers |
| archive/ | cnn_kepler, cnn_kepler_kaggle, cnn_multimodal, model_colab, scalar_norm | dead experiments |

**Note:** if you retrain via the notebooks, they save loose files here — move the
new `cnn_*_seed*.pt` + `scalar_norm_*.npz` into the matching folder (or add a
PROFILE entry) so serving picks them up.
