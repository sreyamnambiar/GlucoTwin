"""Generate an additive synthetic record with comorbidities, HRV and sleep stages."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate import make_patient, simulate_patient


def extend_patient(p: dict, rng: np.random.Generator) -> dict:
    hypertension = int(rng.random() < (0.18 + 0.012 * max(0, p["age"] - 40)))
    dyslipidemia = int(p["ldl"] >= 130 or rng.random() < 0.18)
    ckd_stage = int(np.select(
        [p["egfr"] < 30, p["egfr"] < 60, p["egfr"] < 90],
        [4, 3, 2], default=1
    )) if rng.random() < 0.8 else 0
    cardiac_event = int(rng.random() < (0.02 + 0.0015 * max(0, p["age"] - 45)
                                        + 0.02 * hypertension))
    return dict(p, hypertension=hypertension, dyslipidemia=dyslipidemia,
                ckd_stage=ckd_stage, prior_cardiac_event=cardiac_event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="data/extended_synthetic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    start = pd.Timestamp("2026-01-01 00:00")
    ehr, cgm_all, meals_all, sleep_all, gt = [], [], [], [], []
    for pid in range(a.patients):
        p = extend_patient(make_patient(pid, rng), rng)
        cgm, meals, sleep, truth = simulate_patient(p, a.days, start, rng)
        hr = cgm["heart_rate"].to_numpy(float)
        diffs = np.diff(hr[np.isfinite(hr)])
        p["rmssd"] = round(float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0, 2)
        sleep = sleep.copy()
        sleep["light_pct"] = (100 - sleep["deep_pct"] - rng.uniform(12, 24, len(sleep))).clip(35, 75).round(1)
        sleep["rem_pct"] = (100 - sleep["deep_pct"] - sleep["light_pct"]).round(1)
        gt.append(truth)
        ehr.append(p)
        cgm_all.append(cgm)
        meals_all.append(meals)
        sleep_all.append(sleep)
    pd.DataFrame(ehr).to_parquet(f"{a.out}/ehr.parquet", index=False)
    pd.concat(cgm_all).to_parquet(f"{a.out}/cgm.parquet", index=False)
    pd.concat(meals_all).to_parquet(f"{a.out}/meals.parquet", index=False)
    pd.concat(sleep_all).to_parquet(f"{a.out}/sleep.parquet", index=False)
    pd.DataFrame(gt).to_parquet(f"{a.out}/ground_truth.parquet", index=False)
    print(f"Wrote extended record: {a.patients} patients x {a.days} days to {a.out}")


if __name__ == "__main__":
    main()
