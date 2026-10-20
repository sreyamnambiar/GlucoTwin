"""
Twin validation: replay each patient's meals chronologically (no peeking).
Before each meal, the twin predicts the 2 h rise; after 2 h it observes the outcome and updates.
Compare: prior-only twin (HbA1c prior, never updated) vs live-updating twin vs constant baseline.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from twin.engine import PatientTwin
from models.spike_model import START, STEP, HORIZON, _nm, _sleep_for, prepare_patient_arrays
from sklearn.metrics import mean_absolute_error, roc_auc_score

d = "data/synthetic"
ehr = pd.read_parquet(f"{d}/ehr.parquet").set_index("patient_id")
cgm = pd.read_parquet(f"{d}/cgm.parquet"); meals = pd.read_parquet(f"{d}/meals.parquet"); sleep = pd.read_parquet(f"{d}/sleep.parquet")
gt = pd.read_parquet(f"{d}/ground_truth.parquet").set_index("patient_id")
rows, finals = [], []
for pid, c in cgm.groupby("patient_id"):
    if pid < 120: continue   # patients 0-119 were used to calibrate OBS_BIAS; evaluate on the rest
    raw, g, hr, st = prepare_patient_arrays(c)
    twin = PatientTwin(ehr.loc[pid].to_dict())
    prior = PatientTwin(ehr.loc[pid].to_dict())
    finals.append((pid, ehr.loc[pid], prior.m, twin))
    m = meals[meals.patient_id == pid].sort_values("timestamp"); sl = sleep[sleep.patient_id == pid]
    for k, (_, r) in enumerate(m.iterrows()):
        i = int((r.timestamp - START).total_seconds() // 60 // STEP)
        if i < 72 or i + HORIZON >= len(raw): continue
        pre = _nm(g[i - 2:i + 1]); w = raw[i:i + HORIZON + 1]
        if not np.isfinite(w).any() or not np.isfinite(pre): continue
        rise = float(np.nanmax(w) - pre)
        walked = float(np.nansum(st[i + 2:i + 9])) >= 150          # inferred from step data (10-45 min after meal)
        sh, _ = _sleep_for(sl, r.timestamp)
        rows.append(dict(pid=pid, k=k, rise=rise, live=twin.predict_rise(r.glycemic_load, walked, sh),
                         prior=prior.predict_rise(r.glycemic_load, walked, sh), n_obs=twin.n_obs, gl=r.glycemic_load))
        twin.observe_meal(r.glycemic_load, rise, walked, sh)         # outcome observed 2 h later -> update
df = pd.DataFrame(rows)
# parameter recovery vs the generator's latent gain (only possible because data is synthetic)
tg, pm, lm, nobs = [], [], [], []
for pid, e, pri_m, tw in finals:
    true = gt.loc[pid, "true_gain"]
    tg.append(np.log(true)); pm.append(pri_m); lm.append(tw.m); nobs.append(tw.n_obs)
tg, pm, lm = map(np.array, (tg, pm, lm))
rec = dict(rmse_log_gain_prior=round(float(np.sqrt(np.mean((pm - tg) ** 2))), 3),
           rmse_log_gain_live=round(float(np.sqrt(np.mean((lm - tg) ** 2))), 3),
           median_abs_pct_err_prior=round(float(np.median(np.abs(np.exp(pm - tg) - 1)) * 100), 1),
           median_abs_pct_err_live=round(float(np.median(np.abs(np.exp(lm - tg) - 1)) * 100), 1),
           mean_meals_observed=round(float(np.mean(nobs)), 1))
print("parameter recovery:", rec)
late = df[df.n_obs >= 10]                                           # after the twin has seen >= 10 meals
res = {}
for name, sub in [("all meals", df), ("after 10 observed meals", late)]:
    res[name] = dict(n=len(sub), mae_prior_twin=round(mean_absolute_error(sub.rise, sub.prior), 1),
                     mae_live_twin=round(mean_absolute_error(sub.rise, sub.live), 1),
                     mae_constant=round(mean_absolute_error(sub.rise, np.full(len(sub), df.rise.mean())), 1),
                     auroc_spike_live=round(roc_auc_score(sub.rise >= 60, sub.live), 3),
                     auroc_spike_prior=round(roc_auc_score(sub.rise >= 60, sub.prior), 3))
    print(name, res[name])
by_n = df.groupby(pd.cut(df.n_obs, [-1, 0, 4, 9, 19, 100])).apply(lambda x: round(mean_absolute_error(x.rise, x.live), 1), include_groups=False)
print("live-twin MAE by #meals observed:\n", by_n.to_string())
res["mae_by_meals_observed"] = {str(k): v for k, v in by_n.items()}
res["parameter_recovery"] = rec
json.dump(res, open("models/artifacts/twin_eval.json", "w"), indent=2)
