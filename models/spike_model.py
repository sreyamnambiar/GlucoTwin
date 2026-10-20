"""
Post-prandial spike model.

At the moment a meal is logged, predict whether glucose will rise by >= SPIKE_RISE mg/dL
(peak within the next 2 h minus the pre-meal level). Only information available at the
meal time is used: recent CGM/HR/steps, the meal itself, previous meal, last night's
sleep, and static EHR features. Planned post-meal walks are NOT a feature (future info).

Split by patient. Compared against: HbA1c-only, glycemic-load-only, and CGM-only baselines.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error

STEP = 5
SPIKE_RISE = 60.0
HORIZON = 24  # 2 h
START = pd.Timestamp("2026-01-01 00:00")
SLOT = {"breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3}
STATIC_NUM = ["age", "bmi", "hba1c", "duration_years", "egfr", "ldl", "triglycerides", "family_history",
              "tcf7l2_risk_alleles", "on_metformin", "adherence", "activity_level", "late_dinner",
              "rest_hr", "usual_sleep_h"]
DIET = {"rice_heavy": 0, "mixed": 1, "millet_leaning": 2}


def _nm(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def meal_feature_row(g, hr, steps, i, meal, prev_meal, sleep_h, deep_pct, ehr):
    """Features for a meal at grid index i. g is forward-filled glucose (np.array)."""
    pre = _nm(g[i - 2:i + 1])
    row = {
        "pre_glucose": pre,
        "g_diff_15m": pre - g[i - 3], "g_diff_30m": pre - g[i - 6],
        "g_mean_1h": _nm(g[i - 12:i + 1]), "g_std_1h": float(np.nanstd(g[i - 12:i + 1])),
        "g_mean_6h": _nm(g[i - 72:i + 1]), "g_max_3h": float(np.nanmax(g[i - 36:i + 1])),
        "hr_30m": _nm(hr[i - 6:i + 1]), "steps_1h": float(np.nansum(steps[i - 12:i + 1])),
        "hour": (i * STEP % 1440) / 60.0,
        "slot": SLOT[meal["slot"]], "carbs_g": meal["carbs_g"], "gi": meal["gi"], "glycemic_load": meal["glycemic_load"],
        "mins_since_prev_meal": prev_meal["mins_ago"] if prev_meal else 600.0,
        "prev_meal_gl": prev_meal["gl"] if prev_meal else 0.0,
        "sleep_hours": sleep_h, "deep_pct": deep_pct,
        "diet_pattern": DIET[ehr["diet_pattern"]], "region_south": int(ehr["region"] == "S"),
        "sex_male": int(ehr["sex"] == "M"),
    }
    for c in STATIC_NUM:
        row[c] = ehr[c]
    return row


def _sleep_for(sleep_df, ts):
    """Last night's sleep is only known after 07:00; before that use the previous night."""
    day = ts.normalize()
    if ts.hour < 7:
        day = day - pd.Timedelta(days=1)
    r = sleep_df[sleep_df["date"] == day.date()]
    return (float(r.sleep_hours.iloc[0]), float(r.deep_pct.iloc[0])) if len(r) else (np.nan, np.nan)


def prepare_patient_arrays(cgm_p):
    c = cgm_p.sort_values("timestamp")
    raw = c["glucose"].to_numpy(float)
    g = pd.Series(raw).ffill(limit=12).to_numpy()
    return raw, g, c["heart_rate"].to_numpy(float), c["steps"].to_numpy(float)


def build_dataset(d):
    ehr = pd.read_parquet(f"{d}/ehr.parquet").set_index("patient_id")
    cgm = pd.read_parquet(f"{d}/cgm.parquet")
    meals = pd.read_parquet(f"{d}/meals.parquet")
    sleep = pd.read_parquet(f"{d}/sleep.parquet")
    rows = []
    for pid, c in cgm.groupby("patient_id"):
        raw, g, hr, st = prepare_patient_arrays(c)
        m = meals[meals.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
        sl = sleep[sleep.patient_id == pid]
        for k, r in m.iterrows():
            i = int((r.timestamp - START).total_seconds() // 60 // STEP)
            if i < 72 or i + HORIZON >= len(raw):
                continue
            window = raw[i:i + HORIZON + 1]
            pre = _nm(g[i - 2:i + 1])
            if not np.isfinite(window).any() or not np.isfinite(pre):
                continue
            prev = None
            if k > 0:
                pm = m.iloc[k - 1]
                prev = {"mins_ago": (r.timestamp - pm.timestamp).total_seconds() / 60, "gl": pm.glycemic_load}
            sh, dp = _sleep_for(sl, r.timestamp)
            row = meal_feature_row(g, hr, st, i, r, prev, sh, dp, ehr.loc[pid])
            peak = float(np.nanmax(window))
            row.update(patient_id=pid, timestamp=r.timestamp, rise=peak - pre, peak=peak,
                       y=int(peak - pre >= SPIKE_RISE))
            rows.append(row)
    return pd.DataFrame(rows)


FEATURES_DROP = {"patient_id", "timestamp", "rise", "peak", "y"}


def feature_cols(df):
    return [c for c in df.columns if c not in FEATURES_DROP]


def reliability(y, p, bins=5):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return t.round(3).reset_index(drop=True).to_dict("records")


def main(d="data/synthetic", out="models/artifacts"):
    os.makedirs(out, exist_ok=True)
    df = build_dataset(d)
    feats = feature_cols(df)
    pids = np.array(sorted(df.patient_id.unique()))
    np.random.default_rng(0).shuffle(pids)
    n = len(pids)
    sp = {"train": pids[:int(.7 * n)], "val": pids[int(.7 * n):int(.85 * n)], "test": pids[int(.85 * n):]}
    S = {k: df[df.patient_id.isin(v)] for k, v in sp.items()}
    print({k: (len(v), v.patient_id.nunique()) for k, v in S.items()}, f"| spike rate={df.y.mean():.2f}")

    kw = dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    cls = lgb.LGBMClassifier(**kw).fit(S["train"][feats], S["train"].y, eval_set=[(S["val"][feats], S["val"].y)],
                                       callbacks=[lgb.early_stopping(50, verbose=False)])
    reg = lgb.LGBMRegressor(**kw).fit(S["train"][feats], S["train"].rise, eval_set=[(S["val"][feats], S["val"].rise)],
                                      callbacks=[lgb.early_stopping(50, verbose=False)])
    te = S["test"].copy()
    te["p"] = cls.predict_proba(te[feats])[:, 1]
    te["pred_rise"] = reg.predict(te[feats])

    def lr(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        m.fit(S["train"][cols].fillna(0), S["train"].y)
        return m.predict_proba(te[cols].fillna(0))[:, 1]

    base = {
        "hba1c_only": lr(["hba1c"]),
        "glycemic_load_only": lr(["glycemic_load"]),
        "cgm_only (pre-glucose + trend)": lr(["pre_glucose", "g_diff_15m", "g_diff_30m", "g_mean_1h"]),
        "meal+hba1c": lr(["glycemic_load", "gi", "hba1c"]),
    }
    res = {"n_test_meals": int(len(te)), "spike_rate": round(float(te.y.mean()), 3)}
    res["fusion_lightgbm"] = dict(auroc=round(roc_auc_score(te.y, te.p), 3), auprc=round(average_precision_score(te.y, te.p), 3),
                                  brier=round(brier_score_loss(te.y, te.p), 3),
                                  rise_mae=round(mean_absolute_error(te.rise, te.pred_rise), 1))
    for k, p in base.items():
        res[k] = dict(auroc=round(roc_auc_score(te.y, p), 3), auprc=round(average_precision_score(te.y, p), 3),
                      brier=round(brier_score_loss(te.y, p), 3))
    res["rise_mae_predict_mean"] = round(mean_absolute_error(te.rise, np.full(len(te), S["train"].rise.mean())), 1)
    res["by_slot_auroc"] = {s: round(roc_auc_score(g.y, g.p), 3) for s, g in te.groupby("slot") if g.y.nunique() > 1}
    res["calibration"] = reliability(te.y.values, te.p.values)
    print(json.dumps({k: v for k, v in res.items() if k != "calibration"}, indent=1))
    print("calibration:", res["calibration"])
    imp = pd.Series(cls.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).head(8).round(3).to_dict())

    cls.booster_.save_model(f"{out}/spike_cls.txt")
    reg.booster_.save_model(f"{out}/spike_reg.txt")
    json.dump({"features": feats, "spike_rise_threshold": SPIKE_RISE, "metrics": res}, open(f"{out}/spike_meta.json", "w"), indent=2)
    df.to_parquet(f"{out}/meal_dataset.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:])
