"""
quickstart.py — first runnable milestone.

Proves the whole toolchain works end-to-end on ONE real star:
  download (NASA MAST) -> clean -> BLS transit search -> phase-fold -> save plot.

Run:  python quickstart.py
"""
import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt
import lightkurve as lk

# WASP-18 — a hot Jupiter with a deep, short-period transit (TIC 100100827,
# P ~ 0.94 d, depth ~0.9%): an easy, unambiguous target to prove the toolchain.
TIC = "TIC 100100827"


def main():
    print(f"1/4  Searching NASA MAST for {TIC} ...")
    search = lk.search_lightcurve(TIC, mission="TESS")
    if len(search) == 0:
        raise SystemExit(f"No TESS light curve found for {TIC}")
    lc = search[0].download()

    print("2/4  Cleaning (normalize -> flatten -> remove outliers) ...")
    lc = lc.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)

    print("3/4  Running BLS transit search ...")
    pg = lc.to_periodogram(method="bls", minimum_period=0.5,
                           maximum_period=5.0, frequency_factor=500)
    period = pg.period_at_max_power
    t0 = pg.transit_time_at_max_power
    depth = pg.depth_at_max_power
    print(f"     best period : {period.value:.4f} d")
    print(f"     mid-transit : {t0.value:.4f}")
    print(f"     depth       : {float(depth)*1e6:.0f} ppm")

    print("4/4  Phase-folding and saving plot ...")
    ax = lc.fold(period=period, epoch_time=t0).scatter(s=2, alpha=0.5)
    ax.set_title(f"{TIC}  |  P = {period.value:.3f} d")
    ax.figure.savefig("quickstart_result.png", dpi=120, bbox_inches="tight")
    print("\nDone. Open quickstart_result.png — if you see a dip, the toolchain works.")


if __name__ == "__main__":
    main()
