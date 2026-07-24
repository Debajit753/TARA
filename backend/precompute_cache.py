"""Precompute the analysis cache for the popular stars list.

Runs the full pipeline once for every target in data/popular_stars.json and
writes the JSON result to data/cache/ — after this, /analyze serves those
targets instantly (and the UI feels ~2 s even including network overhead).

Usage:  cd tara && .venv/bin/python backend/precompute_cache.py [--refresh]
"""
import os, sys, json, time, argparse, signal, socket

socket.setdefaulttimeout(90)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from app.main import run_analysis, _cache_path, _finite, POPULAR_PATH  # noqa: E402


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))


def main(refresh: bool):
    with open(POPULAR_PATH) as f:
        stars = json.load(f)
    print(f"Precomputing {len(stars)} popular targets ...")
    done = skip = fail = 0
    for s in stars:
        tic = s["tic"]
        cp = _cache_path(tic)
        if os.path.exists(cp) and not refresh:
            print(f"skip  {tic:16s} (already cached)")
            skip += 1
            continue
        t0 = time.time()
        signal.alarm(180)   # skip any target that hangs (stuck MAST download)
        try:
            out = _finite(run_analysis(tic))   # same inf/nan scrub the API applies
            with open(cp, "w") as f:
                json.dump(out, f)
            lbl = out["classification"]["label"]
            print(f"ok    {tic:16s} -> {lbl:15s} {time.time()-t0:5.1f}s")
            done += 1
        except _Timeout:
            print(f"FAIL  {tic:16s} timeout >180s")
            fail += 1
        except Exception as e:
            print(f"FAIL  {tic:16s} {type(e).__name__}: {str(e)[:60]}")
            fail += 1
        finally:
            signal.alarm(0)
    print(f"\ncached={done}  skipped={skip}  failed={fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute even if cached")
    main(ap.parse_args().refresh)
