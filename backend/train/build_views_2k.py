"""Local view builder for the 2k CNN dataset (runs on the M4, batch by batch).

Builds global(2001)+local(201) views + kepids + 11 catalog scalars for
1000 planets + 1000 false positives, saving incrementally to
backend/data/processed/kepler_views_2k.npz (same keys as the Colab builder:
G, L, Y, K, SC — upload that file to Colab and train on the T4).

Resume-aware: already-built stars are skipped. Run in batches:
    .venv/bin/python backend/train/build_views_2k.py --limit 50
"""
import os, sys, time, signal, socket, argparse
import numpy as np
import pandas as pd

socket.setdefaulttimeout(90)
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
OUT = os.path.join(BACKEND, "data", "processed", "kepler_views_2k.npz")

SCCOLS = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr",
          "koi_impact", "koi_steff", "koi_slogg", "koi_srad", "koi_teq", "koi_insol"]


class _TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))


def _binned(ph, f, hw, n):
    sel = np.abs(ph) < hw; x, y = ph[sel], f[sel]
    if len(y) < n:
        o = np.argsort(x)
        return np.interp(np.linspace(-hw, hw, n), x[o] if len(x) else [0.0], y[o] if len(y) else [1.0])
    bins = np.linspace(-hw, hw, n + 1); idx = np.clip(np.digitize(x, bins) - 1, 0, n - 1)
    out = np.ones(n)
    for b in range(n):
        v = y[idx == b]
        if len(v): out[b] = np.median(v)
    return out


def _norm(v):
    v = v - np.median(v); mn = np.min(v)
    return v / abs(mn) if mn < 0 else v


def views(kepid, period, t0, dur_hr, ng=2001, nl=201):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"KIC {int(kepid)}", author="Kepler", cadence="long")
    lc = (sr[:4].download_all().stitch().remove_nans().normalize()
             .flatten(window_length=401).remove_outliers(sigma=5))
    t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float)
    ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
    durf = (dur_hr / 24.0) / period
    return _norm(_binned(ph, f, 0.5, ng)), _norm(_binned(ph, f, max(2 * durf, 0.01), nl))


def get_targets(per_class):
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    cols = "kepid,koi_time0bk,koi_disposition," + ",".join(SCCOLS)
    df = NasaExoplanetArchive.query_criteria(table="q1_q17_dr25_koi", select=cols).to_pandas()
    df = df.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])
    df["y"] = np.where(df.koi_disposition == "FALSE POSITIVE", 0,
              np.where(df.koi_disposition.isin(["CONFIRMED", "CANDIDATE"]), 1, -1))
    df = df[df.y >= 0].drop_duplicates("kepid")
    return pd.concat([df[df.y == 0].head(per_class),
                      df[df.y == 1].head(per_class)]).reset_index(drop=True)


def main(per_class, limit, timeout_s):
    print("Fetching KOI targets ...", flush=True)
    samp = get_targets(per_class)
    if os.path.exists(OUT):
        d = np.load(OUT)
        G, L, Y, K, SC = [list(d[k]) for k in ("G", "L", "Y", "K", "SC")]
        done = set(int(x) for x in d["K"])
        print(f"RESUMING: {len(done)} stars already built", flush=True)
    else:
        G, L, Y, K, SC, done = [], [], [], [], [], set()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def save():
        np.savez(OUT, G=np.array(G, dtype="float32"), L=np.array(L, dtype="float32"),
                 Y=np.array(Y), K=np.array(K), SC=np.array(SC, dtype="float32"))

    new, start = 0, time.time()
    try:
        for i, r in samp.iterrows():
            if int(r.kepid) in done: continue
            if new >= limit: break
            signal.alarm(timeout_s)
            try:
                g, l = views(r.kepid, r.koi_period, r.koi_time0bk, r.koi_duration)
                G.append(g); L.append(l); Y.append(int(r.y)); K.append(int(r.kepid))
                SC.append([float(r[c]) if pd.notna(r[c]) else np.nan for c in SCCOLS])
                done.add(int(r.kepid)); new += 1
                print(f"ok   [{i+1}/{len(samp)}] built={len(Y)} (+{new}/{limit})  {time.time()-start:.0f}s", flush=True)
                if len(Y) % 25 == 0:
                    save(); print("  -- checkpoint --", flush=True)
            except _TO:
                print(f"SKIP [{i+1}] KIC {int(r.kepid)}  timeout >{timeout_s}s", flush=True)
            except Exception as e:
                print(f"SKIP [{i+1}] KIC {int(r.kepid)}  {type(e).__name__}: {str(e)[:50]}", flush=True)
            finally:
                signal.alarm(0)
    finally:
        save()
        yarr = np.array(Y)
        print(f"\nSAVED {len(Y)} total (planets {int(yarr.sum()) if len(yarr) else 0}) -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=50, help="new stars to build this run")
    ap.add_argument("--timeout", type=int, default=90)
    a = ap.parse_args()
    main(a.per_class, a.limit, a.timeout)
