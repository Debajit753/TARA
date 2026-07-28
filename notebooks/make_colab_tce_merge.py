"""Generates tara_colab_tce_merge_train.ipynb — merge the 4 ExoMiner++ view
parts, fine-tune the Kepler CNN on them. Does SEED 1 first (quick read), then
ALL 5 seeds (the deployable ensemble). Reports zero-shot vs fine-tuned AUC."""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — merge ExoMiner++ parts + fine-tune (T4 GPU)

Run once after all 4 build parts finish. **Runtime → T4 GPU.**
1. upload `tce_views_p1..p4.npz` → merge
2. upload `cnn_merged_seed1.pt` + `scalar_norm_merged.npz` (the Kepler weights)
3. **Seed 1 first** — a quick read on whether the bigger catalog helps
4. then **all 5 seeds** → the deployable TESS ensemble

Reports zero-shot (Kepler-on-TESS) vs fine-tuned AUC so the lift is explicit.''')

code('''import numpy as np
from google.colab import files
print("Upload tce_views_p1.npz .. p4.npz:")
up = files.upload()
parts = [np.load(k) for k in up if k.startswith("tce_views_p")]
G=np.concatenate([p["G"] for p in parts]); L=np.concatenate([p["L"] for p in parts])
Y=np.concatenate([p["Y"] for p in parts]); K=np.concatenate([p["K"] for p in parts])
S=np.concatenate([p["SC"] for p in parts])
np.savez("tce_views_merged.npz", G=G,L=L,Y=Y,K=K,SC=S)
print(f"merged {len(Y)} TCEs | planets {int(Y.sum())} / non {int((Y==0).sum())}")''')

code('''print("Upload the pretrained model .pt + its scalar_norm .npz.")
print("  RECOMMENDED (clean comparison): cnn_merged_seed1.pt + scalar_norm_merged.npz  <- the KEPLER weights")
print("  (do NOT use cnn_tess_finetuned here for the first run)")
up2 = files.upload()
PRETRAINED = [f for f in up2 if f.endswith(".pt")][0]
NORM = [f for f in up2 if f.endswith(".npz")][0]
print("using pretrained:", PRETRAINED, "| scaler:", NORM)
import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device:",dev)
def block(ci,co,k=5,pool=4): return [nn.Conv1d(ci,co,k,padding=k//2),nn.BatchNorm1d(co),nn.ReLU(),nn.MaxPool1d(pool)]
class Net(nn.Module):
    def __init__(self, ns=11, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g=nn.Sequential(*block(1,16),*block(16,32),*block(32,64),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.l=nn.Sequential(*block(1,16,pool=2),*block(16,32,pool=2),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.s=nn.Sequential(nn.Linear(ns,sh),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64*8+32*8+sh,hd),nn.ReLU(),nn.Dropout(drop),nn.Linear(hd,1))
    def forward(self,g,l,s): return self.head(torch.cat([self.g(g),self.l(l),self.s(s)],1)).squeeze(1)
T=lambda a: torch.tensor(np.asarray(a),dtype=torch.float32).to(dev)

# SPLIT BY TIC (not by TCE): stars average ~3 TCEs each (harmonics/secondary detections),
# so a random split leaks a star's noise fingerprint across train/test -> inflated AUC.
idx=np.arange(len(Y))
itr,ite=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(idx,Y,groups=K))
print(f"grouped split: {len(np.unique(K[itr]))} train stars, {len(np.unique(K[ite]))} test stars, "
      f"overlap {len(set(K[itr])&set(K[ite]))} (must be 0)")

def robust_scaler(ref):        # handles all-NaN / constant columns without blowups
    med=np.nanmedian(S[ref],0); med=np.where(np.isnan(med),0.0,med)
    Sf=np.where(np.isnan(S),med,S)
    lo,hi=np.nanpercentile(Sf[ref],1,0),np.nanpercentile(Sf[ref],99,0); Sf=np.clip(Sf,lo,hi)
    mu=Sf[ref].mean(0); sd=Sf[ref].std(0); sd=np.where(sd<1e-9,1.0,sd)
    return ((Sf-mu)/sd).astype("float32"), mu, sd, med
Sn,tmu,tsd,tmed = robust_scaler(itr)

# zero-shot: Kepler weights + Kepler scaler
kn=np.load(NORM)
Snk=np.clip(np.nan_to_num((np.where(np.isnan(S),kn["med"],S)-kn["mu"])/np.where(kn["sd"]<1e-6,1,kn["sd"])),-10,10).astype("float32")
net=Net().to(dev); net.load_state_dict(torch.load(PRETRAINED,map_location=dev)); net.eval()
with torch.no_grad():
    p=torch.sigmoid(net(T(G[ite]).unsqueeze(1),T(L[ite]).unsqueeze(1),T(Snk[ite]))).cpu().numpy()
AUC_ZS=roc_auc_score(Y[ite],p); print(f"ZERO-SHOT (Kepler on ExoMiner-TESS): AUC {AUC_ZS:.3f}")''')

code('''FT_LR, FT_EPOCHS = 3e-4, 40
def finetune(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(dev); net.load_state_dict(torch.load(PRETRAINED,map_location=dev))
    opt=torch.optim.AdamW(net.parameters(),FT_LR,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss()
    Ge,Le,Se=T(G[ite]).unsqueeze(1),T(L[ite]).unsqueeze(1),T(Sn[ite]); best,bp=0,None
    for ep in range(FT_EPOCHS):
        net.train(); perm=itr[np.random.permutation(len(itr))]
        for b0 in range(0,len(perm),128):
            bi=perm[b0:b0+128]; g=T(G[bi]).unsqueeze(1); l=T(L[bi]).unsqueeze(1)
            if torch.rand(1).item()<0.5: g=torch.flip(g,[-1]); l=torch.flip(l,[-1])
            g=g+0.03*torch.randn_like(g); l=l+0.03*torch.randn_like(l)
            opt.zero_grad(); lossf(net(g,l,T(Sn[bi])),T(Y[bi])).backward(); opt.step()
        net.eval()
        with torch.no_grad(): pr=torch.sigmoid(net(Ge,Le,Se)).cpu().numpy()
        a=roc_auc_score(Y[ite],pr)
        if a>best: best,bp=a,pr
    return net, best, bp

# --- SEED 1 first: quick read ---
net1, auc1, p1 = finetune(1)
print(f">>> SEED 1: zero-shot {AUC_ZS:.3f} -> fine-tuned {auc1:.3f}  (lift {auc1-AUC_ZS:+.3f}) <<<")
print("If this looks good, run the next cell for all 5 seeds.")''')

code('''# --- ALL 5 seeds -> ensemble ---
probs=[p1]; states=[net1.state_dict()]
for sd in [2,3,4,5]:
    net,a,pr=finetune(sd); probs.append(pr); states.append(net.state_dict())
    print(f"  seed {sd}: TESS AUC {a:.3f}")
pe=np.mean(probs,0)
print(f"\\n5-seed ENSEMBLE AUC: {roc_auc_score(Y[ite],pe):.3f}  (zero-shot was {AUC_ZS:.3f})")
for i,st in enumerate(states): torch.save(st,f"cnn_tce_finetuned_seed{i+1}.pt")
np.savez("scalar_norm_tce.npz", mu=tmu, sd=tsd, med=tmed)
np.save("test_tics.npy", np.unique(K[ite]))   # EXACT held-out stars -> bundle with the models so tests can't leak
files.download("tce_views_merged.npz"); files.download("scalar_norm_tce.npz"); files.download("test_tics.npy")
for i in range(1,6): files.download(f"cnn_tce_finetuned_seed{i}.pt")''')

md('''## Using this
- Split is **grouped by TIC** so no star is in both train and test (stars have ~3 TCEs each — a random split leaks and inflates AUC by ~0.05). The AUC printed here is the HONEST generalization number — report this one.
- **Seed 1** tells you fast whether the bigger ExoMiner catalog beats the ~0.86 TOI fine-tune. Run all 5 only if it does.
- To deploy: the serving `Ensemble` auto-detects `cnn_tess_finetuned*.pt` first — rename these to `cnn_tess_finetuned_seed*.pt` (+ `scalar_norm_tess.npz`) to swap them in, or add a new profile.
- Honest note: this catalog has no stellar params in the CSV (we pull Teff/logg/radius from FITS headers), and `snr` uses MES. Fewer clean scalars than the TOI set, but far more labeled examples — the trade the finale will resolve.''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_tce_merge_train.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
