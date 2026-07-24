"""Dev check — run clean -> search -> features on one star and print the features.
   python dev_check.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from app.pipeline.preprocess import load_and_clean
from app.pipeline.search import bls_search
from app.pipeline.features import extract_features

TIC = "TIC 100100827"  # WASP-18 (deep transit)

print(f"Loading + cleaning {TIC} ...")
lc = load_and_clean(TIC)
print("BLS search ...")
cand = bls_search(lc)
print("Candidate:", {k: round(v, 4) for k, v in cand.items()})
feats, gv, lv = extract_features(lc, cand)
print("\nFeatures:")
for k, v in feats.items():
    print(f"  {k:16s} {v}")
print(f"\nglobal_view: {len(gv)} bins   local_view: {len(lv)} bins")
print("min flux in local view:", round(float(lv.min()), 5))
