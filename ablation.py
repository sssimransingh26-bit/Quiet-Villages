"""Which feature group adds signal? Gradient boosting, same district-held-out folds, same label."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from build_v2 import SIZE, AMENITY, CENSUS, make_gbm, oof
from build_and_baselines import score_block
df = pd.read_csv(sys.argv[1])
y = df.depop.values
sets = {"SIZE": SIZE, "SIZE+AMENITIES": SIZE + AMENITY, "SIZE+CENSUS_SHARES": SIZE + CENSUS, "FULL": SIZE + AMENITY + CENSUS}
out = {}
for k, cols in sets.items():
    b = score_block(y, oof(df, make_gbm, cols)); out[k] = b
    print(f"{k:20s} n_feat={len(cols):2d} AUC {b['auc']:.3f} [{b['auc_ci'][0]:.3f},{b['auc_ci'][1]:.3f}]  P@10% {b['precision_top10pct']:.3f} [{b['ptop10_ci'][0]:.3f},{b['ptop10_ci'][1]:.3f}]")
json.dump(out, open("ablation_gbm.json", "w"), indent=2)
