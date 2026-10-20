"""Fusion ablation: what does each data stream add? (patient-level split, same as spike_model)"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
sys.path.insert(0, ".")
from models.spike_model import STATIC_NUM, feature_cols

df = pd.read_parquet("models/artifacts/meal_dataset.parquet")
MEAL = ["slot", "carbs_g", "gi", "glycemic_load", "hour", "mins_since_prev_meal", "prev_meal_gl"]
STATIC = STATIC_NUM + ["diet_pattern", "region_south", "sex_male"]
DYN = ["pre_glucose", "g_diff_15m", "g_diff_30m", "g_mean_1h", "g_std_1h", "g_mean_6h", "g_max_3h",
       "hr_30m", "steps_1h", "sleep_hours", "deep_pct"]
sets = {"meal only": MEAL, "meal + static EHR": MEAL + STATIC, "meal + dynamic wearables/CGM": MEAL + DYN,
        "FUSION (meal + EHR + wearables/CGM)": MEAL + STATIC + DYN}
pids = np.array(sorted(df.patient_id.unique())); np.random.default_rng(0).shuffle(pids); n = len(pids)
tr, va, te = [df[df.patient_id.isin(x)] for x in (pids[:int(.7*n)], pids[int(.7*n):int(.85*n)], pids[int(.85*n):])]
out = {}
for name, cols in sets.items():
    a, m = [], []
    for seed in range(5):  # average over seeds for stability
        kw = dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
        c = lgb.LGBMClassifier(**kw).fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
        r = lgb.LGBMRegressor(**kw).fit(tr[cols], tr.rise, eval_set=[(va[cols], va.rise)], callbacks=[lgb.early_stopping(50, verbose=False)])
        a.append(roc_auc_score(te.y, c.predict_proba(te[cols])[:, 1])); m.append(mean_absolute_error(te.rise, r.predict(te[cols])))
    out[name] = dict(auroc=round(float(np.mean(a)), 3), rise_mae=round(float(np.mean(m)), 1))
    print(f"{name:40s} AUROC={out[name]['auroc']}  rise MAE={out[name]['rise_mae']}")
json.dump(out, open("models/artifacts/ablation.json", "w"), indent=2)
