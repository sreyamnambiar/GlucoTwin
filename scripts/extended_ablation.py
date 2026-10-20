"""Report a separate synthetic ablation for the extended record."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_rows(root):
    ehr = pd.read_parquet(os.path.join(root, "ehr.parquet")).set_index("patient_id")
    cgm = pd.read_parquet(os.path.join(root, "cgm.parquet"))
    meals = pd.read_parquet(os.path.join(root, "meals.parquet"))
    sleep = pd.read_parquet(os.path.join(root, "sleep.parquet"))
    rows = []
    for pid, ms in meals.groupby("patient_id"):
        g = cgm[cgm.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            continue
        e = ehr.loc[pid]
        s = sleep[sleep.patient_id == pid].set_index("date")
        for _, meal in ms.iterrows():
            i = int(np.searchsorted(g.timestamp.values, meal.timestamp, side="left"))
            if i >= len(g) - 24:
                continue
            pre = g.glucose.iloc[max(0, i - 3):i + 1].to_numpy(float)
            post = g.glucose.iloc[i:i + 25].to_numpy(float)
            if not np.isfinite(pre).any() or not np.isfinite(post).any():
                continue
            label = int(np.nanmax(post) - np.nanmean(pre) >= 60)
            date_row = s.loc[meal.timestamp.date()] if meal.timestamp.date() in s.index else None
            rows.append(dict(
                patient_id=pid, carbs=meal.carbs_g, gi=meal.gi, hba1c=e.hba1c,
                glucose=float(g.glucose.iloc[i]), trend=float(g.glucose.iloc[i] - g.glucose.iloc[max(0, i - 3)]),
                hypertension=e.hypertension, dyslipidemia=e.dyslipidemia, ckd_stage=e.ckd_stage,
                cardiac=e.prior_cardiac_event, rmssd=e.rmssd,
                light_pct=0 if date_row is None else date_row.light_pct,
                deep_pct=0 if date_row is None else date_row.deep_pct,
                rem_pct=0 if date_row is None else date_row.rem_pct, label=label))
    return pd.DataFrame(rows)


def auc(df, features, train_ids, test_ids):
    train, test = df[df.patient_id.isin(train_ids)], df[df.patient_id.isin(test_ids)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    model.fit(train[features], train.label)
    return float(roc_auc_score(test.label, model.predict_proba(test[features])[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/extended_synthetic")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = build_rows(a.root)
    df = df.dropna(subset=["carbs", "gi", "hba1c", "glucose", "trend"]).reset_index(drop=True)
    ids = np.sort(df.patient_id.unique())
    split = max(1, int(len(ids) * 0.8))
    base = ["carbs", "gi", "hba1c", "glucose", "trend"]
    extra = base + ["hypertension", "dyslipidemia", "ckd_stage", "cardiac", "rmssd",
                    "light_pct", "deep_pct", "rem_pct"]
    result = dict(rows=int(len(df)), spike_rate=round(float(df.label.mean()), 4),
                  baseline_auroc=round(auc(df, base, ids[:split], ids[split:]), 4),
                  extended_auroc=round(auc(df, extra, ids[:split], ids[split:]), 4),
                  note="Separate synthetic ablation; it does not replace or alter the production LightGBM model.")
    out = a.out or os.path.join(a.root, "ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
