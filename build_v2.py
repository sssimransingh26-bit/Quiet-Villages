"""Quiet Villages v2: add 2001 census amenities and population shares to the table, score baselines.

Feature list is FIXED here, before any model is scored. Every feature is from the 2001 census.
Two sets are compared:  SIZE (village size/density/neighbourhood only)  vs  FULL (SIZE + amenities + census shares).
Evaluation: whole districts held out (5 folds), same folds as v1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_and_baselines import FEATURES as SIZE, build, folds, score_block, SEED  # noqa: E402

AMENITY = ["has_primary_school", "has_middle_school", "has_secondary_school", "has_medical_facility", "has_hospital",
           "has_health_centre", "has_tap_water", "has_handpump", "has_post_office", "has_bank", "has_bus",
           "has_paved_road", "has_mud_road", "footpath_only", "log_dist_town_km", "has_power", "has_phone",
           "irrigated_share", "forest_share"]
CENSUS = ["literacy", "female_literacy", "child_share", "sc_share", "st_share", "female_per_male", "worker_share",
          "cultivator_share", "ag_labour_share", "other_worker_share", "marginal_worker_share", "household_size"]
FULL = SIZE + AMENITY + CENSUS


def div(a, b):
    return a / b.replace(0, np.nan)


def add_features(df: pd.DataFrame, pca: pd.DataFrame, vd: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(vd, on="shrid2", how="left").merge(pca, on="shrid2", how="left", suffixes=("", "_pca"))
    f = lambda c: (m[c] > 0).astype(float).where(m[c].notna())
    m["has_primary_school"] = f("pc01_vd_p_sch")
    m["has_middle_school"] = f("pc01_vd_m_sch")
    m["has_secondary_school"] = f("pc01_vd_s_sch")
    m["has_medical_facility"] = f("pc01_vd_medi_fac")
    m["has_hospital"] = f("pc01_vd_hosp")
    m["has_health_centre"] = ((m.pc01_vd_phs_cnt > 0) | (m.pc01_vd_ph_cntr > 0)).astype(float).where(m.pc01_vd_phs_cnt.notna())
    m["has_tap_water"] = f("pc01_vd_tap")
    m["has_handpump"] = f("pc01_vd_handpump")
    m["has_post_office"] = f("pc01_vd_post_off")
    m["has_bank"] = f("pc01_vd_bank_fac")
    m["has_bus"] = f("pc01_vd_bs_fac")
    m["has_paved_road"] = f("pc01_vd_app_pr")
    m["has_mud_road"] = f("pc01_vd_app_mr")
    m["footpath_only"] = ((m.pc01_vd_app_fp == 1) & (m.pc01_vd_app_pr == 0) & (m.pc01_vd_app_mr == 0)).astype(float).where(m.pc01_vd_app_fp.notna())
    m["log_dist_town_km"] = np.log1p(m.pc01_vd_dist_town.clip(0, 300))
    m["has_power"] = f("pc01_vd_power_supl")
    m["has_phone"] = f("pc01_vd_phone")
    area = m.pc01_vd_area.replace(0, np.nan)
    m["irrigated_share"] = (m.pc01_vd_tot_irr / area).clip(0, 1)
    m["forest_share"] = (m.pc01_vd_land_fores / area).clip(0, 1)
    tp = m.pc01_pca_tot_p
    m["literacy"] = div(m.pc01_pca_p_lit, tp)
    m["female_literacy"] = div(m.pc01_pca_f_lit, m.pc01_pca_tot_f)
    m["child_share"] = div(m.pc01_pca_p_06, tp)
    m["sc_share"] = div(m.pc01_pca_p_sc, tp)
    m["st_share"] = div(m.pc01_pca_p_st, tp)
    m["female_per_male"] = div(m.pc01_pca_tot_f, m.pc01_pca_tot_m)
    m["worker_share"] = div(m.pc01_pca_tot_work_p, tp)
    w = m.pc01_pca_tot_work_p
    m["cultivator_share"] = div(m.pc01_pca_main_cl_p, w)
    m["ag_labour_share"] = div(m.pc01_pca_main_al_p, w)
    m["other_worker_share"] = div(m.pc01_pca_main_ot_p, w)
    m["marginal_worker_share"] = div(m.pc01_pca_margwork_p, w)
    m["household_size"] = div(tp, m.pc01_pca_no_hh)
    for c in CENSUS:
        m[c] = m[c].clip(upper=m[c].quantile(0.995)) if c == "household_size" else m[c]
    return m


def make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))


def make_gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)


def oof(df, make, cols):
    y, X, out = df.depop.values, df[cols].values.astype(float), np.zeros(len(df))
    for k in sorted(df.fold.unique()):
        tr, te = df.fold.values != k, df.fold.values == k
        mdl = make(); mdl.fit(X[tr], y[tr]); out[te] = mdl.predict_proba(X[te])[:, 1]
    return out


def main(keys_dir, pca_csv, vd_csv, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = build(Path(keys_dir)); df["fold"] = folds(df)
    pca = pd.read_csv(pca_csv, low_memory=False); vd = pd.read_csv(vd_csv, low_memory=False)
    pca = pca[[c for c in pca.columns if c == "shrid2" or c.startswith("pc01_pca_")]]
    vd = vd[[c for c in vd.columns if c == "shrid2" or c.startswith("pc01_vd_")]]
    df = add_features(df, pca, vd)
    y = df.depop.values
    res = {"n": int(len(df)), "positives": int(y.sum()), "prevalence": float(y.mean()),
           "n_features": {"SIZE": len(SIZE), "FULL": len(FULL)}, "models": {}}
    res["models"]["rule_smaller_is_riskier"] = score_block(y, -df.log_pop01.values)
    for tag, cols in (("SIZE", SIZE), ("FULL", FULL)):
        s = oof(df, make_logreg, cols); df[f"oof_logreg_{tag}"] = s; res["models"][f"logistic_regression_{tag}"] = score_block(y, s)
        s = oof(df, make_gbm, cols); df[f"oof_gbm_{tag}"] = s; res["models"][f"gradient_boosting_{tag}"] = score_block(y, s)
    # which features matter: permutation importance of the FULL gradient-boosting model on held-out districts
    imp = np.zeros(len(FULL))
    for k in sorted(df.fold.unique()):
        tr, te = df.fold.values != k, df.fold.values == k
        mdl = make_gbm().fit(df.loc[tr, FULL].values.astype(float), y[tr])
        r = permutation_importance(mdl, df.loc[te, FULL].values.astype(float), y[te], scoring="roc_auc", n_repeats=5, random_state=SEED)
        imp += r.importances_mean
    imp /= df.fold.nunique()
    res["gbm_full_permutation_importance_auc_drop"] = {c: round(float(v), 4) for c, v in sorted(zip(FULL, imp), key=lambda t: -t[1])}
    # simple descriptive view: depopulation rate by amenity
    desc = {}
    for c in ["has_paved_road", "has_primary_school", "has_medical_facility", "has_power", "has_bus", "has_tap_water"]:
        g = df.groupby(c).depop.agg(["mean", "size"])
        desc[c] = {str(int(k)): {"depop_rate": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in g.iterrows()}
    res["depop_rate_by_amenity"] = desc
    df.to_csv(out / "uk_villages_v2.csv", index=False)
    (out / "results_v2_baselines.json").write_text(json.dumps(res, indent=2))
    (out / "feature_sets.json").write_text(json.dumps({"SIZE": SIZE, "FULL": FULL}, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
