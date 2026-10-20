"""Physiological plausibility checks for the synthetic cohort."""
import sys
import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
ehr = pd.read_parquet(f"{d}/ehr.parquet")
cgm = pd.read_parquet(f"{d}/cgm.parquet")
meals = pd.read_parquet(f"{d}/meals.parquet")

g = cgm.groupby("patient_id")["glucose"]
pat = ehr.set_index("patient_id").join(pd.DataFrame({
    "mean_g": g.mean(), "sd_g": g.std(),
    "tir": g.apply(lambda x: ((x >= 70) & (x <= 180)).mean() * 100),
    "tar": g.apply(lambda x: (x > 180).mean() * 100),
    "tbr": g.apply(lambda x: (x < 70).mean() * 100),
}))
pat["eag"] = 28.7 * pat["hba1c"] - 46.7
pat["gmi"] = 3.31 + 0.02392 * pat["mean_g"]  # glucose management indicator

print("=== Cohort ===")
print(f"patients={len(pat)}  readings={len(cgm):,}  meals={len(meals):,}")
print(f"HbA1c mean={pat.hba1c.mean():.2f}  age mean={pat.age.mean():.1f}  BMI mean={pat.bmi.mean():.1f}")

print("\n=== Does CGM agree with HbA1c? (should be close) ===")
print(f"mean CGM glucose={pat.mean_g.mean():.0f} mg/dL vs mean eAG={pat.eag.mean():.0f}")
print(f"corr(HbA1c, mean CGM glucose)={np.corrcoef(pat.hba1c, pat.mean_g)[0,1]:.2f}")
print(f"mean |GMI - HbA1c|={np.abs(pat.gmi - pat.hba1c).mean():.2f}")

print("\n=== Glycemic metrics (typical T2D: TIR 40-70%, TBR <4%) ===")
print(f"TIR(70-180)={pat.tir.mean():.0f}%  TAR(>180)={pat.tar.mean():.0f}%  TBR(<70)={pat.tbr.mean():.1f}%")
print(f"within-patient SD mean={pat.sd_g.mean():.0f} mg/dL  CV={(pat.sd_g/pat.mean_g).mean()*100:.0f}%")

# post-meal rise: peak in 3h after meal minus pre-meal value
cgm = cgm.set_index(["patient_id", "timestamp"]).sort_index()
rises = []
for _, r in meals.sample(min(3000, len(meals)), random_state=0).iterrows():
    s = cgm.loc[r.patient_id].glucose
    pre = s.loc[r.timestamp - pd.Timedelta(minutes=10):r.timestamp].mean()
    post = s.loc[r.timestamp:r.timestamp + pd.Timedelta(hours=3)]
    if len(post) and not np.isnan(pre):
        rises.append((r.slot, r.glycemic_load, post.max() - pre, post.max(), r.walked_after))
rs = pd.DataFrame(rises, columns=["slot", "gl", "rise", "peak", "walked"])
print("\n=== Post-meal response ===")
print(f"median rise={rs.rise.median():.0f}  p90={rs.rise.quantile(.9):.0f} mg/dL (T2D typical 40-100)")
print(f"peak>180 after {((rs.peak>180).mean()*100):.0f}% of meals")
print(rs.groupby("slot").rise.median().round(0).to_string())
print(f"walked vs not (median rise): {rs[rs.walked==1].rise.median():.0f} vs {rs[rs.walked==0].rise.median():.0f}")
print(f"corr(glycemic load, rise)={rs[['gl','rise']].corr().iloc[0,1]:.2f}")

print("\n=== Diet pattern -> mean glucose, controlling for HbA1c ===")
pat["resid"] = pat.mean_g - np.poly1d(np.polyfit(pat.hba1c, pat.mean_g, 1))(pat.hba1c)
print(pat.groupby("diet_pattern").resid.mean().round(1).to_string())
