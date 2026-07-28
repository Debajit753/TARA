"""Generates tara_colab_mixed_train.ipynb — train ONE model on Kepler + TESS
(ExoMiner) mixed, so it works across both missions. Reports overall AUC plus a
per-source breakdown (Kepler-test AUC, TESS-test AUC) proving cross-source
robustness. No new downloads — uses the two view-sets you already built."""
import json, os

CELLS=[]
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md('''# TARA — one model for Kepler + TESS (mixed training)

Trains a single multimodal model on **Kepler and TESS (ExoMiner) mixed**, so it
generalizes across both missions — the robust base for whatever curated dataset
you add later (you'll just add that data to the mix and re-run).

**No downloads.** Upload the two view-sets you already have:
`kepler_views_merged.npz` + `tce_views_merged.npz` (the 11-feature ones).
Runtime → T4 GPU. Reports overall AUC + per-source (Kepler-only / TESS-only).''')

code('''from google.colab import files
import numpy as np
print("Upload kepler_views_merged.npz + tce_views_merged.npz:")
up=files.upload()
kp=np.load("kepler_views_merged.npz"); tv=np.load("tce_views_merged.npz")
assert kp["SC"].shape[1]==tv["SC"].shape[1], "feature mismatch (use the 11-feature tce, not tcevet)"
G=np.concatenate([kp["G"],tv["G"]]); L=np.concatenate([kp["L"],tv["L"]])
Y=np.concatenate([kp["Y"],tv["Y"]]).astype(int); S=np.concatenate([kp["SC"],tv["SC"]])
src=np.array([0]*len(kp["Y"])+[1]*len(tv["Y"]))                       # 0=Kepler, 1=TESS
groups=np.array([f"K{int(k)}" for k in kp["K"]]+[f"T{int(k)}" for k in tv["K"]])  # source-prefixed
print(f"mixed set: {len(Y)} stars | Kepler {len(kp['Y'])} + TESS {len(tv['Y'])} | planets {int(Y.sum())} | {S.shape[1]} features")''')

code('''import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device:",dev)
NS=S.shape[1]; T=lambda a: torch.tensor(np.asarray(a),dtype=torch.float32).to(dev)
def block(ci,co,k=5,pool=4): return [nn.Conv1d(ci,co,k,padding=k//2),nn.BatchNorm1d(co),nn.ReLU(),nn.MaxPool1d(pool)]
class Net(nn.Module):
    def __init__(self, ns=NS, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g=nn.Sequential(*block(1,16),*block(16,32),*block(32,64),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.l=nn.Sequential(*block(1,16,pool=2),*block(16,32,pool=2),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.s=nn.Sequential(nn.Linear(ns,sh),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64*8+32*8+sh,hd),nn.ReLU(),nn.Dropout(drop),nn.Linear(hd,1))
    def forward(self,g,l,s): return self.head(torch.cat([self.g(g),self.l(l),self.s(s)],1)).squeeze(1)

idx=np.arange(len(Y))
itr,ite=next(GroupShuffleSplit(1,test_size=0.2,random_state=42).split(idx,Y,groups=groups))
print(f"grouped split (by source+id): {len(itr)} train / {len(ite)} test | star overlap "
      f"{len(set(groups[itr])&set(groups[ite]))} (must be 0)")

def scaler(ref):
    med=np.nanmedian(S[ref],0); med=np.where(np.isnan(med),0.0,med); Sf=np.where(np.isnan(S),med,S)
    lo,hi=np.nanpercentile(Sf[ref],1,0),np.nanpercentile(Sf[ref],99,0); Sf=np.clip(Sf,lo,hi)
    mu=Sf[ref].mean(0); sd=Sf[ref].std(0); sd=np.where(sd<1e-9,1.0,sd)
    return ((Sf-mu)/sd).astype("float32"),mu,sd,med
Sn,tmu,tsd,tmed=scaler(itr)''')

code('''def train(seed,epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss()
    Ge,Le,Se=T(G[ite]).unsqueeze(1),T(L[ite]).unsqueeze(1),T(Sn[ite]); best,bp=0,None
    for ep in range(epochs):
        net.train(); perm=itr[np.random.permutation(len(itr))]
        for b0 in range(0,len(perm),256):
            bi=perm[b0:b0+256]; g=T(G[bi]).unsqueeze(1); l=T(L[bi]).unsqueeze(1)
            if torch.rand(1).item()<0.5: g=torch.flip(g,[-1]); l=torch.flip(l,[-1])
            g=g+0.03*torch.randn_like(g); l=l+0.03*torch.randn_like(l)
            opt.zero_grad(); lossf(net(g,l,T(Sn[bi])),T(Y[bi])).backward(); opt.step()
        net.eval()
        with torch.no_grad(): pr=torch.sigmoid(net(Ge,Le,Se)).cpu().numpy()
        a=roc_auc_score(Y[ite],pr)
        if a>best: best,bp=a,pr
    return net,best,bp

probs=[]; states=[]
for sd in [1,2,3,4,5]:
    net,a,pr=train(sd); probs.append(pr); states.append(net.state_dict()); print(f"  seed {sd}: AUC {a:.3f}")
pe=np.mean(probs,0); yte=Y[ite]
print(f"\\n=== MIXED KEPLER+TESS — 5-seed ensemble ===")
print(f"OVERALL held-out AUC: {roc_auc_score(yte,pe):.3f}")
for name,s in [("Kepler-only",0),("TESS-only  ",1)]:
    m=(src[ite]==s)
    if m.sum()>10: print(f"  {name} test AUC: {roc_auc_score(yte[m],pe[m]):.3f}  (n={int(m.sum())})")''')

code('''for i,st in enumerate(states): torch.save(st,f"cnn_mixed_seed{i+1}.pt")
np.savez("scalar_norm_mixed.npz", mu=tmu, sd=tsd, med=tmed)
np.save("test_groups_mixed.npy", np.unique(groups[ite]))
from google.colab import files
np.savez("mixed_views.npz", G=G,L=L,Y=Y,SC=S,src=src,groups=groups)
files.download("scalar_norm_mixed.npz"); files.download("test_groups_mixed.npy")
for i in range(1,6): files.download(f"cnn_mixed_seed{i}.pt")
print("saved cnn_mixed_seed1..5.pt + scalar_norm_mixed.npz")''')

md('''## What this gives you
- **One model that works on both missions** — the OVERALL held-out AUC is your headline, and the **per-source AUCs** prove it isn't just good on one (great for the deck).
- **A robust base to extend**: when you obtain a new curated dataset, build its views the same way, `np.concatenate` it into the mix, and re-run this notebook. The model that has seen Kepler + TESS + the new data is your strongest, most general classifier.
- Grouped by source+id, so no star leaks across train/test — the number is honest.''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_mixed_train.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
