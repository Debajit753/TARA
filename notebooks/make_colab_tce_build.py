"""Generates tara_colab_tce_build_split.ipynb — build views from the ExoMiner++
LABELED TESS catalog (8.9 MB, sectors 1-67), split into 4 parts for parallel
building across up to 4 Colab accounts. Then merge+train with the companion
notebook.

Catalog cols: uid, target_id(TIC), label, tce_period, tce_duration(DAYS),
tce_time0bk(BTJD), ruwe, tce_prad, tce_max_mult_ev(MES~SNR), score, fold.
No stellar params in the CSV -> we read Teff/logg/radius from each FITS header."""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — ExoMiner++ labeled TESS catalog, build views (split into 4)

Builds global/local views + scalars from the ExoMiner++ **labeled** catalog
(TESS sectors 1-67). Run on **up to 4 Colab accounts in parallel**:
- Account A: `PART = 1`   ·   B: `PART = 2`   ·   C: `PART = 3`   ·   D: `PART = 4`

**Upload the catalog CSV you already downloaded** when prompted (same file on
each account). Each builds a non-overlapping quarter and downloads
`tce_views_p{PART}.npz`. CPU runtime is fine. When all 4 are done, run the
**merge-train** notebook.''')

code('''PART = 1          # <-- set to 2 / 3 / 4 on the other accounts
N_PARTS = 4
CAP_PER_CLASS = 4000   # cap per class before splitting (uses all if fewer exist)
!pip install -q lightkurve''')

code('''import numpy as np, pandas as pd, os
from google.colab import files
# use an already-uploaded CSV if present, else prompt to upload it
csv_name = next((f for f in os.listdir(".") if f.endswith(".csv") and "labeled_tces" in f), None)
if csv_name is None:
    print("Upload the ExoMiner++ labeled catalog CSV (the 8.9 MB file you downloaded):")
    up = files.upload()
    csv_name = [k for k in up if k.endswith(".csv")][0]
print("using catalog:", csv_name)
df = pd.read_csv(csv_name)
df = df.dropna(subset=["target_id","tce_period","tce_time0bk","tce_duration"])
print("label vocabulary:", df.label.value_counts().to_dict())

PLANET    = {"KP","CP"}                                   # known / confirmed planets
NONPLANET = {"FP","EB","BEB","NEB","NPC","IS","V","O","BD","AFP","NTP"}   # false positives
df["y"] = np.where(df.label.isin(PLANET), 1, np.where(df.label.isin(NONPLANET), 0, -1))
unmapped = sorted(set(df.label.dropna().unique()) - PLANET - NONPLANET)
if unmapped: print("UNMAPPED labels (dropped -> PC/UNK/etc):", unmapped)
df = df[df.y >= 0]

pos, neg = df[df.y==1], df[df.y==0]
n = min(len(pos), len(neg), CAP_PER_CLASS)
full = pd.concat([pos.head(n), neg.head(n)]).reset_index(drop=True)
samp = full.iloc[(PART-1)::N_PARTS].reset_index(drop=True)   # interleaved, non-overlapping quarter
print(f"labeled usable: planets {len(pos)}, non {len(neg)} -> balanced {n}/class")
print(f"PART {PART}/{N_PARTS}: building {len(samp)}", samp.y.value_counts().to_dict())''')

code('''import lightkurve as lk, signal, time
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(_TO()))

def _binned(ph, f, hw, nb):
    sel=np.abs(ph)<hw; x,y=ph[sel],f[sel]
    if len(y)<nb:
        o=np.argsort(x); return np.interp(np.linspace(-hw,hw,nb), x[o] if len(x) else [0.0], y[o] if len(y) else [1.0])
    bins=np.linspace(-hw,hw,nb+1); idx=np.clip(np.digitize(x,bins)-1,0,nb-1); out=np.ones(nb)
    for b in range(nb):
        v=y[idx==b]
        if len(v): out[b]=np.median(v)
    return out
def _norm(v): v=v-np.median(v); mn=np.min(v); return v/abs(mn) if mn<0 else v

def build_one(tic, period, t0, dur_days):     # t0 is BTJD already; dur in days
    lcr = lk.search_lightcurve(f"TIC {int(tic)}", mission="TESS")[0].download()
    meta = lcr.meta
    lc = lcr.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)
    t=np.asarray(lc.time.value,float); f=np.asarray(lc.flux.value,float)
    ph=((t-t0+0.5*period)%period)/period-0.5
    durf=dur_days/period
    g=_norm(_binned(ph,f,0.5,2001)); l=_norm(_binned(ph,f,max(2*durf,0.01),201))
    intr=np.abs(ph)<0.5*durf; oot=np.abs(ph)>1.5*durf
    depth_ppm=(np.median(f[oot])-np.median(f[intr]))*1e6 if intr.sum()>3 else np.nan
    return g, l, depth_ppm, meta

# 11-slot scalars to match the pretrained model:
# [period, duration_hr, depth_ppm, prad, snr(MES), impact, steff, slogg, srad, teq, insol]
OUT=f"tce_views_p{PART}.npz"; G,L,Y,K,SC=[],[],[],[],[]
def save(): np.savez(OUT, G=np.array(G,dtype="float32"), L=np.array(L,dtype="float32"),
                     Y=np.array(Y), K=np.array(K), SC=np.array(SC,dtype="float32"))
start=time.time()
for i,r in samp.iterrows():
    signal.alarm(90)
    try:
        g,l,depth,meta=build_one(r.target_id, r.tce_period, r.tce_time0bk, r.tce_duration)
        sc=[r.tce_period, r.tce_duration*24.0, depth, r.tce_prad, r.tce_max_mult_ev, np.nan,
            meta.get("TEFF",np.nan), meta.get("LOGG",np.nan), meta.get("RADIUS",np.nan), np.nan, np.nan]
        G.append(g); L.append(l); Y.append(int(r.y)); K.append(int(r.target_id)); SC.append(sc)
        if len(Y)%25==0: save(); print(f"  built {len(Y)}/{len(samp)}  {(time.time()-start)/60:.0f} min")
    except (_TO, Exception) as e:
        print(f"skip TIC {int(r.target_id)}: {type(e).__name__}")
    finally:
        signal.alarm(0)
save(); print(f"DONE part {PART}: {len(Y)} stars |", int(np.sum(Y)), "planets")''')

code('''from google.colab import files
files.download(f"tce_views_p{PART}.npz")''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_tce_build_split.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
