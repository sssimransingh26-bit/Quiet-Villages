"""Quiet Villages: build the Uttarakhand 2001->2011 table, label it, score baselines.

Inputs : SHRUG Population Census keys (pc01r_shrid_key.dta, pc11r_shrid_key.dta)
Outputs: uk_villages.csv (features + label + fold) and baselines.json

Rules fixed BEFORE any model was scored:
  population 2001 >= 50
  depopulated = 2011 population < 70% of 2001, OR no 2011 row for that unit
  features use 2001 information only (no 2011 column is ever a feature)
  evaluation = whole districts held out (GroupKFold over district), never random village splits
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATE = "05"
MIN_POP = 50
RATIO = 0.7
SEED = 20261031
FEATURES = ["log_pop01", "land_area01", "log_density01", "sub_mean_logpop", "sub_median_logdens", "sub_n_units"]


def build(keys_dir: Path) -> pd.DataFrame:
    a = pd.read_stata(keys_dir / "pc01r_shrid_key.dta", convert_categoricals=False)
    b = pd.read_stata(keys_dir / "pc11r_shrid_key.dta", convert_categoricals=False)
    a = a[a.pc01_state_id == STATE].copy()
    b = b[b.pc11_state_id == STATE].copy()
    # a few unit ids repeat: merge them by summing (documented in the post)
    a = a.groupby("shrid2", as_index=False).agg(
        district=("pc01_district_id", "first"), subdistrict=("pc01_subdistrict_id", "first"),
        pop01=("pc01_pca_tot_p", "sum"), land_area01=("pc01_land_area", "sum"))
    b = b.groupby("shrid2", as_index=False).agg(pop11=("pc11_pca_tot_p", "sum"))
    n_all = len(a)
    a = a[a.shrid2.str[:2] == "11"]          # 2001 units that carry a 2011-style id (can be linked)
    n_linkable = len(a)
    a = a[a.pop01 >= MIN_POP]
    m = a.merge(b, on="shrid2", how="left")
    m["no_2011_row"] = m.pop11.isna().astype(int)
    m["ratio"] = m.pop11 / m.pop01
    m["depop"] = ((m.ratio < RATIO) | m.pop11.isna()).astype(int)
    m["depop_strict"] = np.where(m.pop11.isna(), np.nan, (m.ratio < RATIO).astype(float))
    # 2001-only features
    m["log_pop01"] = np.log(m.pop01)
    m["land_area01"] = m.land_area01.clip(lower=0.01)
    m["density01"] = m.pop01 / m.land_area01
    m["log_density01"] = np.log(m.density01)
    g = m.groupby(["district", "subdistrict"])
    m["sub_mean_logpop"] = g.log_pop01.transform("mean")
    m["sub_median_logdens"] = g.log_density01.transform("median")
    m["sub_n_units"] = g.log_pop01.transform("count")
    m.attrs["counts"] = dict(units_2001=n_all, linkable=n_linkable, kept_pop50=len(m))
    return m


def folds(df: pd.DataFrame) -> np.ndarray:
    fold = np.zeros(len(df), dtype=int)
    for i, (_, te) in enumerate(GroupKFold(n_splits=5).split(df, groups=df.district)):
        fold[te] = i
    return fold


def top_decile_precision(y, s, frac=0.10):
    k = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-s)[:k]
    return float(np.mean(y[idx]))


def boot_ci(y, s, fn, n=500, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def score_block(y, s):
    return {
        "auc": float(roc_auc_score(y, s)), "auc_ci": boot_ci(y, s, roc_auc_score),
        "avg_precision": float(average_precision_score(y, s)), "ap_ci": boot_ci(y, s, average_precision_score),
        "precision_top10pct": top_decile_precision(y, s),
        "ptop10_ci": boot_ci(y, s, top_decile_precision),
    }


def out_of_fold(df, make_model, cols):
    y = df.depop.values
    X = df[cols].values
    oof = np.zeros(len(df))
    for f in sorted(df.fold.unique()):
        tr, te = df.fold.values != f, df.fold.values == f
        mdl = make_model()
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return oof


def main(keys_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build(Path(keys_dir))
    df["fold"] = folds(df)
    df.to_csv(out / "uk_villages.csv", index=False)
    y = df.depop.values
    res = {"counts": df.attrs["counts"], "n": int(len(df)), "positives": int(y.sum()),
           "prevalence": float(y.mean()), "no_2011_row": int(df.no_2011_row.sum()),
           "districts": int(df.district.nunique()), "models": {}}
    # baseline 1: "small villages empty out" - the plain rule
    res["models"]["rule_smaller_is_riskier"] = score_block(y, -df.log_pop01.values)
    # baseline 2: logistic regression on the 2001 village features
    oof = out_of_fold(df, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), FEATURES)
    res["models"]["logistic_regression"] = score_block(y, oof)
    df["oof_logreg"] = oof
    # baseline 3: gradient boosting
    oof = out_of_fold(df, lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED), FEATURES)
    res["models"]["gradient_boosting"] = score_block(y, oof)
    df["oof_gbm"] = oof
    # the same rule without the unmatched units (sensitivity check)
    ok = df.depop_strict.notna().values
    ys = df.depop_strict.values[ok].astype(int)
    res["models"]["rule_matched_only"] = score_block(ys, -df.log_pop01.values[ok])
    res["sensitivity_matched_only"] = {"n": int(ok.sum()), "prevalence": float(ys.mean())}
    df.to_csv(out / "uk_villages_with_baseline_scores.csv", index=False)
    (out / "baselines.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
