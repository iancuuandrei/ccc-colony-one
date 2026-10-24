from pathlib import Path
import numpy as np
from train import load
files=sorted(Path("data/ext_site_tactics_v1").glob("chunk*.npz"))+sorted(Path("data/ext_site_v1_rs").glob("chunk*.npz"))+sorted(Path("data/ext_site_gen05_losses_rs").glob("chunk*.npz"))
D=load(files)
mix=0.5*D["pval"].astype(np.float32)+0.5*D["pval_g"].astype(np.float32)
bad=np.abs(mix-D["pval"].astype(np.float32)).sum(1)
print("rows",len(D["z"]),"overrides",D.get("_overrides_applied"),"has_length ext",int(D["has_length"].sum()),"max mix deviation on all rows", float(bad.max()))
