"""Stage 2 — search for periodic transit-like dips (BLS and TLS)."""
import os
import signal
import numpy as np


def bls_search(lc, min_period=0.5, max_period=12.0):
    """Box Least Squares — fast, coarse duration. Good fallback."""
    pg = lc.to_periodogram(method="bls", minimum_period=min_period,
                           maximum_period=max_period, frequency_factor=500)
    return {
        "period": float(pg.period_at_max_power.value),
        "t0": float(pg.transit_time_at_max_power.value),
        "depth": float(pg.depth_at_max_power),
        "duration": float(pg.duration_at_max_power.value),
        "power": float(pg.max_power.value),
    }


def fast_search(lc, min_period=0.5, max_period=10.0):
    """Fast two-stage search: coarse BLS on a binned curve (~0.5 s) to find the
    period, then TLS refined in a narrow ±3% window around it (~0.5-1 s).
    Returns (candidate, periodogram_data) — the BLS periodogram is reused for
    the UI chart so it is never computed twice."""
    try:
        blc = lc.bin(time_bin_size=10.0 / 1440.0)   # 10-min bins -> ~5x fewer points
    except Exception:
        blc = lc
    pg = blc.to_periodogram(method="bls", minimum_period=min_period,
                            maximum_period=max_period, frequency_factor=300)
    p0 = float(pg.period_at_max_power.value)
    lo = max(min_period, p0 * 0.97)
    hi = min(max_period * 1.2, p0 * 1.03)
    try:
        cand = tls_search(lc, min_period=lo, max_period=hi)   # narrow -> fast
    except Exception:
        cand = {   # BLS-only fallback
            "period": p0,
            "t0": float(pg.transit_time_at_max_power.value),
            "depth": float(pg.depth_at_max_power),
            "duration": float(pg.duration_at_max_power.value),
            "power": float(pg.max_power.value),
        }
    pgdata = {"p": pg.period.value, "power": pg.power.value, "best": cand["period"]}
    return cand, pgdata


def tls_search(lc, min_period=0.5, max_period=10.0):
    """Transit Least Squares — fits a realistic limb-darkened transit shape,
    giving a much sharper period & duration than BLS (key for clean depth and
    V-vs-U-shape features). Tuned for speed in bulk dataset building:
    narrower period range + lighter grids (~1.5-2x faster, tiny precision cost)."""
    from transitleastsquares import transitleastsquares
    t = np.ascontiguousarray(lc.time.value, dtype="float64")
    f = np.ascontiguousarray(lc.flux.value, dtype="float64")
    m = np.isfinite(t) & np.isfinite(f)
    res = transitleastsquares(t[m], f[m]).power(
        period_min=min_period, period_max=max_period,
        oversampling_factor=2, duration_grid_step=1.15,
        use_threads=os.cpu_count() or 2, show_progress_bar=False)
    return {
        "period": float(res.period),
        "t0": float(res.T0),
        "depth": float(abs(1.0 - res.depth)),
        "duration": float(res.duration),
        "power": float(res.SDE),
        "period_uncertainty": float(getattr(res, "period_uncertainty", float("nan"))),
    }


def deep_search(lc, cap_seconds=180, min_period=0.5, max_period=15.0):
    """THOROUGH full-range TLS for hard stars — for shallow candidates where the
    fast two-stage search locks a period alias (e.g. TOI-1022: fast finds 4.93 d,
    the truth is 3.097 d). Unlike fast_search, this evaluates the WHOLE period grid
    with TLS's proper transit statistics, so it can recover the real period even
    when it isn't the tallest BLS peak.

    Wall-clock capped: runs in a separate PROCESS and returns None if it can't
    finish in `cap_seconds` (the caller then keeps the fast result). ~60-120 s on
    a cool machine; on a thermally-throttled fanless laptop it will hit the cap.

    A process, not a thread: TLS itself fans out to a worker pool, and a timed-out
    thread cannot be killed — every timeout used to leave (cpu_count-2) CPU-bound
    workers running to completion, starving the normal fast path. The child is put
    in its own process group so the timeout can kill the whole tree, TLS pool
    included.
    """
    import multiprocessing as mp
    import queue as _queue

    t = np.ascontiguousarray(lc.time.value, dtype="float64")
    f = np.ascontiguousarray(lc.flux.value, dtype="float64")
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 50:
        return None

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    nthreads = max(1, (os.cpu_count() or 2) - 2)
    # NOT daemon=True: a daemonic process may not create children, and TLS fans
    # out to its own pool. Cleanup is explicit via _kill_tree() instead.
    p = ctx.Process(target=_deep_worker,
                    args=(t, f, min_period, max_period, nthreads, q), daemon=False)
    p.start()
    try:
        ok, payload = q.get(timeout=cap_seconds)
    except (_queue.Empty, Exception):
        ok, payload = False, None
    finally:
        _kill_tree(p)
    return payload if ok else None


def _kill_tree(p):
    """Kill the deep-search child and anything it spawned, then reap it."""
    try:
        if p.is_alive():
            killed = False
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    killed = True
                except (OSError, AttributeError):
                    killed = False
            if not killed:
                p.terminate()          # Windows / no process groups
        p.join(5)
    except Exception:
        pass


def _deep_worker(t, f, pmin, pmax, nthreads, q):
    """Module-level so it is picklable under the 'spawn' start method."""
    try:
        os.setsid()      # own process group -> parent can kill the whole tree
    except (OSError, AttributeError):
        pass
    try:
        from transitleastsquares import transitleastsquares
        res = transitleastsquares(t, f).power(
            period_min=pmin, period_max=pmax,
            oversampling_factor=3, duration_grid_step=1.05,   # denser = more thorough
            use_threads=nthreads, show_progress_bar=False)
        q.put((True, {
            "period": float(res.period), "t0": float(res.T0),
            "depth": float(abs(1.0 - res.depth)), "duration": float(res.duration),
            "power": float(res.SDE),
            "period_uncertainty": float(getattr(res, "period_uncertainty", float("nan"))),
        }))
    except Exception as e:
        q.put((False, f"{type(e).__name__}: {e}"))
