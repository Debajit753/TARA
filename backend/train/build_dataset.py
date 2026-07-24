"""Phase 2 — build a labelled feature dataset for training (robust version).

Pulls TESS Objects of Interest (TOI) + dispositions from the NASA Exoplanet Archive,
downloads a balanced subset of light curves, extracts features, and writes a labelled
table to backend/data/processed/dataset.csv.

Robustness:
  - per-target timeout (a hung MAST download is skipped, not fatal)
  - incremental CSV save (each row written as it completes — no progress lost)
  - flushed, live progress output

Disposition -> class:
  KP, CP -> transit | FP -> false_positive | FA -> noise | PC -> skipped

Usage:  python backend/train/build_dataset.py --per-class 40 [--timeout 90]
"""
import os, sys, argparse, signal, socket
import pandas as pd

socket.setdefaulttimeout(90)  # global network safety net

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
from app.pipeline.preprocess import load_raw, clean_lc
from app.pipeline.search import bls_search, tls_search
from app.pipeline.features import extract_features
from app.pipeline.blend import centroid_shift

DISP_MAP = {"KP": "transit", "CP": "transit", "FP": "false_positive", "FA": "noise"}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def get_labelled_targets(per_class):
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    toi = NasaExoplanetArchive.query_criteria(table="toi", select="tid,tfopwg_disp")
    df = toi.to_pandas().dropna(subset=["tfopwg_disp"])
    df["label"] = df["tfopwg_disp"].map(DISP_MAP)
    df = df.dropna(subset=["label"]).drop_duplicates("tid")
    picks = [g.head(per_class) for _, g in df.groupby("label")]
    return pd.concat(picks)[["tid", "label"]].reset_index(drop=True)


def process_one(tic):
    raw = load_raw(tic)             # raw LC keeps centroid columns
    lc = clean_lc(raw)
    try:
        cand = tls_search(lc)       # sharper period + duration
    except Exception:
        cand = bls_search(lc)       # fallback if TLS fails
    feats, _, _ = extract_features(lc, cand)
    feats["centroid"] = centroid_shift(raw, cand)   # Phase 4: blend check (~1 ms)
    return feats


def main(per_class, timeout_s):
    print("Fetching TOI labels from NASA Exoplanet Archive ...", flush=True)
    targets = get_labelled_targets(per_class)
    print(f"Selected {len(targets)} targets "
          f"({targets['label'].value_counts().to_dict()})\n", flush=True)

    out_dir = os.path.join(BACKEND, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.csv")

    written = 0
    header_done = False
    with open(path, "w") as f:
        for i, r in targets.iterrows():
            tic = f"TIC {int(r.tid)}"
            signal.alarm(timeout_s)
            try:
                feats = process_one(tic)
                feats["tic"] = tic
                feats["label"] = r.label
                pd.DataFrame([feats]).to_csv(f, index=False, header=not header_done)
                f.flush()
                header_done = True
                written += 1
                print(f"OK   [{i+1}/{len(targets)}] {tic:16s} {r.label}  ({written} saved)", flush=True)
            except _Timeout:
                print(f"SKIP [{i+1}/{len(targets)}] {tic:16s} timeout >{timeout_s}s", flush=True)
            except Exception as e:
                print(f"SKIP [{i+1}/{len(targets)}] {tic:16s} {type(e).__name__}: {str(e)[:70]}", flush=True)
            finally:
                signal.alarm(0)

    print(f"\nDone. {written} rows -> {path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--timeout", type=int, default=90)
    a = p.parse_args()
    main(a.per_class, a.timeout)
