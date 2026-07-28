"""Generates tara_colab_sector_run.ipynb — the Stage-3 'apply to a real TESS
sector' pipeline. Downloads real light curves, runs search+classify on each, and
writes ONE result row per star (process-and-discard: the light curve is thrown
away immediately, so disk never fills). 4,000 stars split into 2 parts (2k each)
across two accounts. CPU-bound (TLS search) — GPU optional. Resumable: re-running
skips stars already in the results CSV."""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — apply to a real TESS sector (Stage 3)

Runs the full pipeline (search → classify) on real, never-seen TESS stars — the
finale's core requirement. **Process-and-discard:** each light curve is downloaded,
turned into one result row, then deleted, so disk never fills (no 45 GB of FITS).

Run on **two accounts**: `PART = 1` and `PART = 2` (2,000 stars each = 4,000 total).
CPU runtime is fine (the TLS search is CPU-bound; GPU only speeds the tiny model).
**Resumable** — if it disconnects, just re-run; it skips stars already done.

Upload the model files when asked: `cnn_tce_finetuned_seed1..5.pt` + `scalar_norm_tce.npz`.''')

code('''PART = 1          # <-- set to 2 on the second account
N_PARTS = 2
SECTOR = 1        # which TESS sector to apply to
N_TOTAL = 4000    # stars across both parts
!pip install -q lightkurve transitleastsquares''')

code('''import numpy as np
from astroquery.mast import Observations
# list the 2-min (SPOC) targets observed in this sector, then take our slice
obs = Observations.query_criteria(obs_collection="TESS", provenance_name="SPOC",
                                  dataproduct_type="timeseries", sequence_number=SECTOR)
tics=[]
for nm in obs["target_name"]:
    s="".join(ch for ch in str(nm) if ch.isdigit())
    if s: tics.append(int(s))
tics=sorted(set(tics))[:N_TOTAL]
mine=tics[(PART-1)::N_PARTS]          # interleaved, non-overlapping half
print(f"sector {SECTOR}: {len(tics)} targets total | PART {PART} handles {len(mine)}")''')

code('''from google.colab import files
print("Upload cnn_tce_finetuned_seed1..5.pt + scalar_norm_tce.npz:")
up=files.upload()
import torch, torch.nn as nn, glob
def block(ci,co,k=5,pool=4): return [nn.Conv1d(ci,co,k,padding=k//2),nn.BatchNorm1d(co),nn.ReLU(),nn.MaxPool1d(pool)]
class Net(nn.Module):
    def __init__(self, ns=11, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g=nn.Sequential(*block(1,16),*block(16,32),*block(32,64),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.l=nn.Sequential(*block(1,16,pool=2),*block(16,32,pool=2),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.s=nn.Sequential(nn.Linear(ns,sh),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64*8+32*8+sh,hd),nn.ReLU(),nn.Dropout(drop),nn.Linear(hd,1))
    def forward(self,g,l,s): return self.head(torch.cat([self.g(g),self.l(l),self.s(s)],1)).squeeze(1)
NETS=[]
for f in sorted(glob.glob("cnn_tce_finetuned_seed*.pt")):
    n=Net(); n.load_state_dict(torch.load(f,map_location="cpu")); n.eval(); NETS.append(n)
NZ=np.load("scalar_norm_tce.npz"); MU,SD,MED=NZ["mu"],NZ["sd"],NZ["med"]
print(f"loaded {len(NETS)}-model ensemble")''')

code('''import lightkurve as lk
from transitleastsquares import transitleastsquares

def _binned(ph,f,hw,n):
    sel=np.abs(ph)<hw; x,y=ph[sel],f[sel]
    if len(y)<n:
        o=np.argsort(x); return np.interp(np.linspace(-hw,hw,n),x[o] if len(x) else [0.],y[o] if len(y) else [1.])
    bins=np.linspace(-hw,hw,n+1); idx=np.clip(np.digitize(x,bins)-1,0,n-1); out=np.ones(n)
    for b in range(n):
        v=y[idx==b]
        if len(v): out[b]=np.median(v)
    return out
def _norm(v): v=v-np.median(v); mn=np.min(v); return v/abs(mn) if mn<0 else v

def analyze(tic):
    lcr=lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", sector=SECTOR)[0].download()
    meta=lcr.meta
    lc=lcr.remove_nans().normalize().flatten(window_length=401).remove_outliers(sigma=5)
    t=np.ascontiguousarray(lc.time.value,float); f=np.ascontiguousarray(lc.flux.value,float)
    m=np.isfinite(t)&np.isfinite(f)
    bls=lc.to_periodogram(method="bls",minimum_period=0.5,maximum_period=12,frequency_factor=500)
    p0=float(bls.period_at_max_power.value)
    r=transitleastsquares(t[m],f[m]).power(period_min=max(0.5,p0*0.9),period_max=p0*1.1,
        oversampling_factor=2,duration_grid_step=1.15,use_threads=2,show_progress_bar=False)
    P,t0,dur,sde=float(r.period),float(r.T0),float(r.duration),float(r.SDE)
    ph=((t-t0+0.5*P)%P)/P-0.5; durf=dur/P
    intr=np.abs(ph)<0.5*durf; oot=np.abs(ph)>1.5*durf
    depth=float(np.median(f[oot])-np.median(f[intr])) if intr.sum()>3 else float(abs(1-r.depth))
    sig=np.std(f[oot]) if oot.sum()>10 else np.std(f); snr=depth/(sig/np.sqrt(max(intr.sum(),1))+1e-12)
    g=_norm(_binned(ph,f,0.5,2001)).astype("float32"); l=_norm(_binned(ph,f,max(2*durf,0.01),201)).astype("float32")
    srad=meta.get("RADIUS",np.nan)
    prad=(np.sqrt(max(depth,0))*float(srad)*109.2) if srad==srad and srad else np.nan
    sc=np.array([P,dur*24,depth*1e6,prad,snr,np.nan,meta.get("TEFF",np.nan),
                 meta.get("LOGG",np.nan),srad,np.nan,np.nan],float)
    sn=np.clip(np.nan_to_num((np.where(np.isnan(sc),MED,sc)-MU)/np.where(SD<1e-9,1,SD)),-10,10).astype("float32")
    gt=torch.tensor(g).view(1,1,-1); lt=torch.tensor(l).view(1,1,-1); st=torch.tensor(sn).view(1,-1)
    with torch.no_grad(): prob=float(np.mean([torch.sigmoid(n(gt,lt,st)).item() for n in NETS]))
    return dict(tic=tic,period=round(P,5),depth_ppm=round(depth*1e6,1),duration_hr=round(dur*24,3),
                snr=round(snr,1),sde=round(sde,1),planet_prob=round(prob,4),
                verdict="planet" if prob>0.45 else "not_planet")
print("pipeline ready")''')

code('''import csv, os, signal, time
OUT=f"sector{SECTOR}_results_p{PART}.csv"
FIELDS=["tic","period","depth_ppm","duration_hr","snr","sde","planet_prob","verdict"]
done=set()
if os.path.exists(OUT):
    for row in csv.DictReader(open(OUT)): done.add(int(row["tic"]))
    print(f"resuming — {len(done)} already done")
fh=open(OUT,"a",newline=""); w=csv.DictWriter(fh,fieldnames=FIELDS)
if not done: w.writeheader()
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(_TO()))
todo=[t for t in mine if t not in done]; start=time.time(); ok=0
for i,tic in enumerate(todo):
    signal.alarm(120)
    try:
        w.writerow(analyze(tic)); fh.flush(); ok+=1
        if ok%20==0: print(f"  {ok}/{len(todo)} done  {(time.time()-start)/60:.0f} min")
    except (_TO,Exception) as e:
        print(f"skip TIC {tic}: {type(e).__name__}")
    finally:
        signal.alarm(0)
fh.close(); print(f"DONE part {PART}: {ok} new rows -> {OUT}")''')

code('''from google.colab import files
files.download(f"sector{SECTOR}_results_p{PART}.csv")''')

md('''## Notes
- **Process-and-discard**: each curve is downloaded, reduced to one row, then dropped — disk stays near-empty, so this scales to a full 20-30k sector.
- **Resumable**: if Colab disconnects, re-run this notebook — it reads the existing CSV and skips finished stars. For a full sector, run in chunks (raise `N_TOTAL`) or mount Drive to keep the CSV across sessions.
- **CPU-bound**: the TLS search is the cost (~2-4 s/star); ~2,000 stars ≈ 1-2 hrs per part. GPU not required.
- Next: run **tara_colab_sector_analyze.ipynb** on the two result CSVs to merge, cross-match known planets, and get the ranked candidate list.''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_sector_run.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
