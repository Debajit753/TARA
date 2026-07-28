"""Generates tara_colab_sector_analyze.ipynb — merge the two sector-run CSVs,
cross-match against known TESS planets (recovery validation), and produce the
ranked candidate list + a summary figure. Small network call for the TOI list only."""
import json, os

CELLS=[]
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md('''# TARA — sector results: merge, validate, candidate list

Upload the two `sector*_results_p1.csv` + `p2.csv` from the run notebook. This
merges them, cross-matches against the **known TESS TOI dispositions** (so you can
say "recovered N of the M known planets in this sector"), and writes the ranked
candidate list. Only one small network call (the TOI catalog CSV).''')

code('''from google.colab import files
import pandas as pd, numpy as np
print("Upload the two sector result CSVs (p1 and p2):")
up=files.upload()
df=pd.concat([pd.read_csv(k) for k in up if k.endswith(".csv")]).drop_duplicates("tic").reset_index(drop=True)
print(f"merged {len(df)} stars | flagged planets (prob>0.45): {(df.planet_prob>0.45).sum()}")''')

code('''# known dispositions for cross-match (recovery check)
TOI="https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+tid,tfopwg_disp+from+toi&format=csv"
toi=pd.read_csv(TOI).dropna(subset=["tid"])
known={int(r.tid): r.tfopwg_disp for _,r in toi.iterrows()}
df["known_disp"]=df.tic.map(known)
is_known_planet=df.known_disp.isin(["KP","CP"])
n_known=int(is_known_planet.sum())
recovered=int(((df.planet_prob>0.45)&is_known_planet).sum())
print(f"known planets (KP/CP) present in our set: {n_known}")
print(f"  of those, we flagged as planet: {recovered}  (recovery {recovered/max(n_known,1):.0%})")
known_fp=df.known_disp.isin(["FP","FA"])
if known_fp.sum():
    rej=int(((df.planet_prob<=0.45)&known_fp).sum())
    print(f"known false-positives present: {int(known_fp.sum())} | correctly rejected: {rej} ({rej/int(known_fp.sum()):.0%})")''')

code('''# ranked candidate list — the actual deliverable
cand=df.sort_values("planet_prob",ascending=False)
new=cand[cand.known_disp.isna() & (cand.planet_prob>0.9)]   # high-confidence, not already catalogued
print(f"TOP 20 CANDIDATES in the sector:")
cols=["tic","planet_prob","period","depth_ppm","duration_hr","snr","sde","known_disp"]
print(cand[cols].head(20).to_string(index=False))
print(f"\\nhigh-confidence signals NOT in the TOI catalog (prob>0.9): {len(new)}")
df.to_csv("sector_merged_results.csv",index=False)
cand[cols].head(200).to_csv("sector_candidate_list.csv",index=False)''')

code('''import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].hist(df.planet_prob,bins=40,color="#4f46e5"); ax[0].set_title("planet-probability distribution")
ax[0].set_xlabel("P(planet)"); ax[0].set_ylabel("stars")
kp=df[df.known_disp.isin(["KP","CP"])]; fp=df[df.known_disp.isin(["FP","FA"])]
ax[1].hist(fp.planet_prob,bins=20,alpha=.6,label="known FP",color="#b45309")
ax[1].hist(kp.planet_prob,bins=20,alpha=.6,label="known planet",color="#0f766e")
ax[1].set_title("known planets vs false-positives"); ax[1].set_xlabel("P(planet)"); ax[1].legend()
plt.tight_layout(); plt.savefig("sector_summary.png",dpi=140); plt.show()
from google.colab import files
files.download("sector_merged_results.csv"); files.download("sector_candidate_list.csv"); files.download("sector_summary.png")''')

md('''## What you get
- **sector_merged_results.csv** — every star with its detected params + planet probability + known disposition.
- **sector_candidate_list.csv** — top 200 ranked by planet probability (the deliverable table).
- **sector_summary.png** — probability distribution + known-planets-vs-false-positives separation.
- The printed **recovery %** ("flagged N of M known planets") is your validation on real data — the strongest possible "it works on unseen stars" evidence.''')

nb={"cells":CELLS,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"},"colab":{"provenance":[]}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tara_colab_sector_analyze.ipynb")
with open(out,"w") as f: json.dump(nb,f,indent=1)
print("wrote",out)
