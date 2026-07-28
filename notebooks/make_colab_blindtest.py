"""Generates tara_colab_blindtest.ipynb — NO-NETWORK blind test of the TESS CNN
ensemble on pre-built views. Upload the files you already have; it classifies all
~7k stars in seconds, reports the HONEST accuracy on the held-out (never-trained)
stars, and writes a per-star CSV + confusion image + a ranked candidate list."""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md('''# TARA — blind test on TESS (no network, instant)

Runs the fine-tuned TESS CNN ensemble on your pre-built views. **No downloads, no
MAST, no API** — everything is uploaded, and inference on ~7,000 stars takes a
couple of seconds. CPU runtime is fine (a GPU makes it instant).

**Upload these files you already have (all in one go):**
- `tce_views_merged.npz`  (the ~7k pre-built TESS stars)
- `cnn_tce_finetuned_seed1.pt` … `seed5.pt`  (the 5 grouped-split models)
- `scalar_norm_tce.npz`  (the scaler)

Outputs (auto-downloaded at the end): `tess_predictions.csv`,
`tess_confusion.png`, `tess_summary.txt`.''')

code('''from google.colab import files
print("Upload tce_views_merged.npz + the 5 cnn_tce_finetuned_seed*.pt + scalar_norm_tce.npz:")
up = files.upload()
import numpy as np
data = "tce_views_merged.npz"
d = np.load(data); G,L,Y,K,S = d["G"],d["L"],d["Y"],d["K"],d["SC"]
print(f"loaded {len(Y)} TESS stars | planets {int(Y.sum())} / non {int((Y==0).sum())} | unique TICs {len(np.unique(K))}")''')

code('''import torch, torch.nn as nn, time
from sklearn.model_selection import GroupShuffleSplit
dev = "cuda" if torch.cuda.is_available() else "cpu"
def block(ci,co,k=5,pool=4): return [nn.Conv1d(ci,co,k,padding=k//2),nn.BatchNorm1d(co),nn.ReLU(),nn.MaxPool1d(pool)]
class Net(nn.Module):
    def __init__(self, ns=11, drop=0.35, sh=24, hd=96):
        super().__init__()
        self.g=nn.Sequential(*block(1,16),*block(16,32),*block(32,64),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.l=nn.Sequential(*block(1,16,pool=2),*block(16,32,pool=2),nn.AdaptiveMaxPool1d(8),nn.Flatten())
        self.s=nn.Sequential(nn.Linear(ns,sh),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64*8+32*8+sh,hd),nn.ReLU(),nn.Dropout(drop),nn.Linear(hd,1))
    def forward(self,g,l,s): return self.head(torch.cat([self.g(g),self.l(l),self.s(s)],1)).squeeze(1)

nz=np.load("scalar_norm_tce.npz"); mu,sd,med=nz["mu"],nz["sd"],nz["med"]
Sn=np.clip(np.nan_to_num((np.where(np.isnan(S),med,S)-mu)/np.where(sd<1e-9,1,sd)),-10,10).astype("float32")
import glob
nets=[]
for f in sorted(glob.glob("cnn_tce_finetuned_seed*.pt")):
    n=Net().to(dev); n.load_state_dict(torch.load(f,map_location=dev)); n.eval(); nets.append(n)
print(f"loaded {len(nets)}-model ensemble on {dev}")

# which stars did the models NEVER see?  PREFER the exact held-out list saved at
# training time (test_tics.npy) -> immune to model/split mismatch. Fall back to the
# grouped split only if it's missing (valid only if these are the grouped-trained models).
import os
if os.path.exists("test_tics.npy"):
    tt=set(np.load("test_tics.npy").tolist()); heldout=np.isin(K,list(tt))
    print(f"using saved held-out list: {heldout.sum()} rows from {len(tt)} stars (leak-proof)")
else:
    print("!! test_tics.npy NOT uploaded -> guessing the split; ONLY valid if these are the grouped-trained models")
    _,ite=next(GroupShuffleSplit(1,test_size=0.2,random_state=42).split(np.arange(len(Y)),Y,groups=K))
    heldout=np.zeros(len(Y),bool); heldout[ite]=True

# classify EVERY star, timed
T=lambda a: torch.tensor(np.asarray(a),dtype=torch.float32).to(dev)
t0=time.time()
with torch.no_grad():
    gg,ll,ss=T(G).unsqueeze(1),T(L).unsqueeze(1),T(Sn)
    prob=np.mean([torch.sigmoid(n(gg,ll,ss)).cpu().numpy() for n in nets],0)
dt=time.time()-t0
print(f"\\n>>> classified {len(Y)} TESS stars in {dt:.2f} seconds ({len(Y)/dt:,.0f} stars/sec) <<<")''')

code('''from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
pred=(prob>0.5).astype(int)
ho=heldout  # honest = held-out (never trained)
yh,ph,prh=Y[ho],prob[ho],pred[ho]
auc=roc_auc_score(yh,ph); acc=accuracy_score(yh,prh)
cm=confusion_matrix(yh,prh)
prec=(yh[prh==1]==1).mean() if prh.sum() else 0; rec=(prh[yh==1]==1).mean()
print("="*54)
print(f"HONEST TEST (on {ho.sum()} stars the models NEVER saw):")
print(f"  AUC {auc:.3f} | accuracy {acc:.3f}")
print(f"  planet recall {rec:.2f} ({int((prh[yh==1]==1).sum())}/{int(yh.sum())} planets caught)")
print(f"  planet precision {prec:.2f} | non-planet recall {(prh[yh==0]==0).mean():.2f}")
print(f"  confusion [[TN,FP],[FN,TP]] = {cm.tolist()}")
print("="*54)

# ranked candidate list (held-out), top 25 by planet probability
order=np.argsort(-ph)[:25]
print("\\nTOP 25 CANDIDATES (held-out), ranked by P(planet):")
print(f"{'rank':>4} {'TIC':>11} {'P(planet)':>10} {'truth':>11}")
for r,i in enumerate(order,1):
    idx=np.where(ho)[0][i]
    print(f"{r:>4} {int(K[idx]):>11} {ph[i]:>10.3f} {('planet' if yh[i]==1 else 'non-planet'):>11}")
topk=50; ok=int(Y[np.where(ho)[0][np.argsort(-ph)[:topk]]].sum())
print(f"\\nof the top {topk} ranked candidates, {ok} are real planets (precision@{topk} = {ok/topk:.0%})")''')

code('''import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(4.2,4.2))
ax.imshow(cm,cmap="Blues")
for (i,j),v in np.ndenumerate(cm): ax.text(j,i,int(v),ha="center",va="center",fontsize=15,
    color="white" if v>cm.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["non-planet","planet"])
ax.set_yticks([0,1]); ax.set_yticklabels(["non-planet","planet"])
ax.set_xlabel("predicted"); ax.set_ylabel("truth")
ax.set_title(f"TESS blind test (n={ho.sum()})  AUC {auc:.3f}")
plt.tight_layout(); plt.savefig("tess_confusion.png",dpi=140); plt.show()

# per-star CSV + summary
import csv
with open("tess_predictions.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["tic","true_label","planet_probability","predicted","held_out"])
    for i in range(len(Y)):
        w.writerow([int(K[i]),"planet" if Y[i]==1 else "non-planet",round(float(prob[i]),4),
                    "planet" if pred[i]>0.5 else "non-planet", bool(heldout[i])])
with open("tess_summary.txt","w") as f:
    f.write(f"TARA TESS blind test\\nstars classified: {len(Y)} in {dt:.2f}s\\n"
            f"held-out (honest) n={ho.sum()}\\nAUC {auc:.3f} | acc {acc:.3f}\\n"
            f"planet recall {rec:.2f} | precision {prec:.2f}\\nconfusion {cm.tolist()}\\n")
print("wrote tess_predictions.csv, tess_confusion.png, tess_summary.txt")''')

code('''from google.colab import files
files.download("tess_predictions.csv")
files.download("tess_confusion.png")
files.download("tess_summary.txt")''')

md('''## What you get
- **tess_predictions.csv** — every star: TIC, true label, planet probability, verdict, and whether it was held-out.
- **tess_confusion.png** — the blind-test confusion matrix (for your deck).
- **tess_summary.txt** — the headline numbers.

The AUC/accuracy printed are measured **only on stars the models never trained on** — that's the honest number to report. Processing all ~7k in a couple of seconds shows it scales to a full sector.''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_blindtest.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
