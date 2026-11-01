"""Quiet Villages - run TabPFN on the same table, same district-held-out folds as the baselines.

Run in Google Colab (GPU) or locally:
    pip install tabpfn scikit-learn pandas numpy
    python run_tabpfn.py uk_villages_with_baseline_scores.csv

Writes: tabpfn_results.json and uk_villages_with_all_scores.csv
Set env FAKE=1 to test the plumbing with a stand-in model (numbers then mean nothing).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_SETS = {"SIZE": ["log_pop01", "land_area01", "log_density01", "sub_mean_logpop", "sub_median_logdens", "sub_n_units"], "FULL": ["log_pop01", "land_area01", "log_density01", "sub_mean_logpop", "sub_median_logdens", "sub_n_units", "has_primary_school", "has_middle_school", "has_secondary_school", "has_medical_facility", "has_hospital", "has_health_centre", "has_tap_water", "has_handpump", "has_post_office", "has_bank", "has_bus", "has_paved_road", "has_mud_road", "footpath_only", "log_dist_town_km", "has_power", "has_phone", "irrigated_share", "forest_share", "literacy", "female_literacy", "child_share", "sc_share", "st_share", "female_per_male", "worker_share", "cultivator_share", "ag_labour_share", "other_worker_share", "marginal_worker_share", "household_size"]}
TRAIN_CAP = 10_000   # TabPFN's standard row limit; training folds are down-sampled (stratified) to this
SEED = 20261031


def make_model():
    if os.environ.get("FAKE") == "1":
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(random_state=SEED)


def top_frac_precision(y, s, frac=0.10):
    k = max(1, int(round(len(y) * frac)))
    return float(np.mean(y[np.argsort(-s)[:k]]))


def boot_ci(y, s, fn, n=500, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def block(y, s):
    return {"auc": float(roc_auc_score(y, s)), "auc_ci": boot_ci(y, s, roc_auc_score),
            "avg_precision": float(average_precision_score(y, s)), "ap_ci": boot_ci(y, s, average_precision_score),
            "precision_top10pct": top_frac_precision(y, s), "ptop10_ci": boot_ci(y, s, top_frac_precision)}


def main(path: str) -> None:
    df = pd.read_csv(path)
    y = df.depop.values
    t0 = time.time()
    results = {"model": "FAKE stand-in" if os.environ.get("FAKE") == "1" else "TabPFN", "train_cap": TRAIN_CAP,
               "n": int(len(df)), "sets": {}}
    for tag, cols in FEATURE_SETS.items():
        X = df[cols].values.astype(float)
        rng = np.random.default_rng(SEED)   # same down-sample for every run
        oof = np.zeros(len(df))
        for f in sorted(df.fold.unique()):
            tr = np.where(df.fold.values != f)[0]
            te = np.where(df.fold.values == f)[0]
            if len(tr) > TRAIN_CAP:   # stratified down-sample
                pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                keep_pos = rng.choice(pos, int(round(TRAIN_CAP * len(pos) / len(tr))), replace=False)
                keep_neg = rng.choice(neg, TRAIN_CAP - len(keep_pos), replace=False)
                tr = np.concatenate([keep_pos, keep_neg])
            mdl = make_model()
            mdl.fit(X[tr], y[tr])
            oof[te] = mdl.predict_proba(X[te])[:, 1]
            print(f"[{tag}] fold {f}: train {len(tr)} test {len(te)}  ({time.time()-t0:.0f}s)", flush=True)
        df[f"oof_tabpfn_{tag}"] = oof
        results["sets"][tag] = block(y, oof)
        print(tag, json.dumps(results["sets"][tag]), flush=True)
    results["seconds"] = round(time.time() - t0)
    df.to_csv("uk_villages_with_all_scores.csv", index=False)
    with open("tabpfn_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n=== FINAL ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
