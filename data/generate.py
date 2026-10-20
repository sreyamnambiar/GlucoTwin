"""
GlucoTwin-IN synthetic data generator.

Generates, per patient:
  - Static EHR  (demographics, HbA1c, labs, TCF7L2 genotype, diet pattern, metformin adherence)
  - Dynamic streams at 5-min resolution: CGM glucose, heart rate, steps
  - Meal log (Indian meal library with carbs / glycemic index)
  - Nightly sleep summary

The two streams are COUPLED: EHR features (HbA1c, BMI, genotype, adherence, diet pattern)
set the patient's insulin resistance, which scales every meal response. Poor sleep and
missed metformin doses raise it further; post-meal walks reduce it.

NOTE: All parameters are literature-inspired assumptions, NOT clinically validated values.
The data supports pipeline validation, not clinical claims.
"""
import argparse
import os
import numpy as np
import pandas as pd

STEP_MIN = 5
PER_DAY = 24 * 60 // STEP_MIN  # 288

# (name, carbs_g, glycemic_index, region)  region: S=South, N=North, all
MEALS = {
    "breakfast": [
        ("idli_sambar", 45, 70, "S"), ("dosa_chutney", 50, 75, "S"), ("poha", 50, 70, "all"),
        ("upma", 48, 65, "all"), ("paratha_curd", 55, 60, "N"), ("ragi_dosa", 40, 55, "S"),
        ("oats_veg", 35, 55, "all"), ("bread_omelette", 35, 70, "all"),
    ],
    "lunch": [
        ("white_rice_sambar", 85, 73, "S"), ("white_rice_dal", 80, 70, "all"),
        ("roti_sabzi", 60, 62, "N"), ("biryani", 95, 75, "all"),
        ("millet_rice_curry", 65, 52, "all"), ("curd_rice", 70, 65, "S"),
        ("rajma_chawal", 90, 68, "N"),
    ],
    "snack": [
        ("tea_biscuits", 25, 65, "all"), ("samosa", 30, 70, "all"), ("sundal", 20, 35, "S"),
        ("fruit", 20, 50, "all"), ("nuts", 8, 25, "all"),
    ],
    "dinner": [
        ("chapati_dal", 55, 58, "all"), ("dosa", 50, 75, "S"), ("white_rice_curry", 80, 73, "all"),
        ("khichdi", 60, 55, "all"), ("idli", 45, 70, "S"), ("roti_paneer", 50, 55, "N"),
        ("millet_roti_sabzi", 45, 50, "all"),
    ],
}
DIET_K = {"rice_heavy": 0.6, "mixed": 0.0, "millet_leaning": -0.6}


def meal_curve(gi: float, n_steps: int = 72) -> np.ndarray:
    """Normalized post-meal excursion shape (peak=1 at tp). Higher GI -> earlier, sharper peak."""
    tp = 90 - 0.5 * gi                      # minutes to peak (~55 for GI 70)
    t = np.arange(n_steps) * STEP_MIN + 1e-6
    x = t / tp
    return x ** 2 * np.exp(2 * (1 - x))     # decays to ~0 by ~3.3*tp


def make_patient(pid: int, rng: np.random.Generator) -> dict:
    age = int(np.clip(rng.normal(52, 11), 25, 78))
    sex = rng.choice(["M", "F"])
    bmi = float(np.clip(rng.normal(27.5, 4.2), 19, 40))
    hba1c = float(np.clip(rng.lognormal(np.log(7.7), 0.17), 6.0, 12.5))
    tcf7l2 = int(rng.choice([0, 1, 2], p=[0.45, 0.42, 0.13]))
    adherence = float(np.clip(rng.beta(6, 2), 0.35, 1.0))
    diet = str(rng.choice(list(DIET_K), p=[0.45, 0.40, 0.15]))
    return dict(
        patient_id=pid, age=age, sex=sex, bmi=round(bmi, 1),
        hba1c=round(hba1c, 1),
        duration_years=int(np.clip(rng.gamma(2.5, 3.0), 0, 30)),
        egfr=round(float(np.clip(rng.normal(88 - 0.5 * (age - 50), 14), 25, 120)), 0),
        ldl=round(float(np.clip(rng.normal(105, 28), 50, 200)), 0),
        triglycerides=round(float(np.clip(rng.normal(165 + 3 * (bmi - 27), 55), 60, 450)), 0),
        family_history=int(rng.random() < 0.55),
        tcf7l2_risk_alleles=tcf7l2,
        on_metformin=int(rng.random() < 0.85),
        adherence=round(adherence, 2),
        diet_pattern=diet,
        region=str(rng.choice(["S", "N"], p=[0.6, 0.4])),
        activity_level=round(float(rng.beta(2, 3)), 2),     # 0..1
        late_dinner=int(rng.random() < 0.4),
        rest_hr=int(np.clip(rng.normal(72, 8), 52, 95)),
        usual_sleep_h=round(float(np.clip(rng.normal(6.8, 0.8), 4.8, 8.5)), 1),
    )


def pick_meal(slot: str, p: dict, rng: np.random.Generator):
    opts = [m for m in MEALS[slot] if m[3] in ("all", p["region"])]
    w = np.array([np.exp(DIET_K[p["diet_pattern"]] * (m[2] - 65) / 10) for m in opts])
    return opts[rng.choice(len(opts), p=w / w.sum())]


def simulate_patient(p: dict, days: int, start: pd.Timestamp, rng: np.random.Generator):
    latent = float(np.exp(rng.normal(0, 0.25)))   # hidden per-patient sensitivity NOT recorded in EHR
    n = days * PER_DAY
    steps = np.zeros(n)
    glucose = np.zeros(n)
    hrs = np.zeros(n)
    meals_rows, sleep_rows = [], []

    eag = 28.7 * p["hba1c"] - 46.7                       # HbA1c -> estimated avg glucose
    base0 = eag - 28                                     # fasting baseline (calibrated in sanity check)
    resistance = (0.70 + 0.15 * (p["hba1c"] - 6.0) + 0.010 * (p["bmi"] - 25)
                  + 0.08 * p["tcf7l2_risk_alleles"] + (0.05 if p["family_history"] else 0))
    amp_scale = 1.45
    resistance_base = resistance
    resistance = resistance * latent

    hours = (np.arange(PER_DAY) * STEP_MIN) / 60.0
    ar = 0.0
    excursion = np.zeros(n + 80)

    for d in range(days):
        sl = slice(d * PER_DAY, (d + 1) * PER_DAY)
        sleep_h = float(np.clip(rng.normal(p["usual_sleep_h"], 0.9), 3.5, 9.5))
        deep_pct = float(np.clip(rng.normal(17 - 1.5 * max(0, 7 - sleep_h), 3), 5, 26))
        sleep_rows.append(dict(patient_id=p["patient_id"], date=(start + pd.Timedelta(days=d)).date(),
                               sleep_hours=round(sleep_h, 2), deep_pct=round(deep_pct, 1)))
        poor_sleep = max(0.0, 7.0 - sleep_h)
        sleep_factor = 1 + 0.10 * poor_sleep
        taken = rng.random() < p["adherence"] if p["on_metformin"] else True
        miss_factor = 1.0 if taken else 1.18
        day_offset = rng.normal(0, 12) + 4 * poor_sleep + (0 if taken else 12)

        # --- activity baseline (steps per 5 min) ---
        day_steps = np.zeros(PER_DAY)
        awake = (hours >= 6.5) & (hours < 22.5)
        lam = 6 + 40 * p["activity_level"]
        day_steps[awake] = rng.poisson(lam, awake.sum())

        # --- meals ---
        slots = [("breakfast", rng.normal(8.0, 0.5)), ("lunch", rng.normal(13.25, 0.5)),
                 ("dinner", rng.normal(20.0 + 1.5 * p["late_dinner"], 0.5))]
        if rng.random() < 0.6:
            slots.append(("snack", rng.normal(17.0, 0.6)))
        for slot, hour in slots:
            idx = int(hour * 60 / STEP_MIN)
            if idx < 0 or idx >= PER_DAY - 1:
                continue
            name, carbs, gi, _ = pick_meal(slot, p, rng)
            portion = float(np.exp(rng.normal(0, 0.25)))
            resp_noise = float(np.exp(rng.normal(0, 0.35)))  # unexplained meal-to-meal response variability
            gl = carbs * portion * gi / 100.0
            walked = rng.random() < (0.10 + 0.45 * p["activity_level"])
            walk_factor = 1.0
            if walked:
                ws = idx + int(rng.integers(2, 7))      # starts 10-30 min after meal
                wl = int(rng.integers(2, 5))            # 10-20 min
                day_steps[ws:ws + wl] += rng.poisson(95, min(wl, PER_DAY - ws))
                walk_factor = 0.70
            amp = amp_scale * gl * resistance * sleep_factor * miss_factor * walk_factor * resp_noise
            curve = meal_curve(gi) * amp
            g_idx = d * PER_DAY + idx
            excursion[g_idx:g_idx + len(curve)] += curve
            if rng.random() < 0.92:   # 8% of meals never logged
              meals_rows.append(dict(
                patient_id=p["patient_id"],
                timestamp=start + pd.Timedelta(minutes=STEP_MIN * g_idx),
                slot=slot, meal=name, carbs_g=round(carbs * portion, 1), gi=gi,
                glycemic_load=round(gl, 1), walked_after=int(walked)))
        for _ in range(int(rng.integers(0, 3))):          # unlogged snacks / stress bumps
            bidx = int(rng.integers(int(8 * 12), int(22 * 12)))
            bamp = float(rng.gamma(2.0, 12.0))
            bc = meal_curve(60) * bamp
            gi0 = d * PER_DAY + bidx
            excursion[gi0:gi0 + len(bc)] += bc
        steps[sl] = day_steps

        # --- baseline + dawn phenomenon ---
        dawn = 12 * np.exp(-((hours - 6.0) / 1.5) ** 2) * (0.5 + p["hba1c"] / 16)
        glucose[sl] = base0 + day_offset + dawn

        # --- heart rate ---
        sleeping = (hours >= 23.0) | (hours < 5.5)
        hr = p["rest_hr"] + 0.30 * (day_steps / STEP_MIN) - 8 * sleeping + 2.0 * poor_sleep
        hrs[sl] = hr + rng.normal(0, 2.5, PER_DAY)

    # --- combine glucose components + AR(1) noise + sensor noise ---
    glucose += excursion[:n]
    noise = np.zeros(n)
    for i in range(n):
        ar = 0.95 * ar + rng.normal(0, 2.4)
        noise[i] = ar
    glucose += noise + rng.normal(0, 2.0, n)
    glucose = np.clip(glucose, 45, 400)

    for _ in range(days * 2):                           # CGM signal-loss gaps (15-75 min)
        a = int(rng.integers(0, n - 20)); glucose[a:a + int(rng.integers(3, 16))] = np.nan
    ts = start + pd.to_timedelta(np.arange(n) * STEP_MIN, unit="m")
    cgm = pd.DataFrame(dict(patient_id=p["patient_id"], timestamp=ts,
                            glucose=glucose.round(1), heart_rate=hrs.round(0), steps=steps.astype(int)))
    return cgm, pd.DataFrame(meals_rows), pd.DataFrame(sleep_rows), dict(patient_id=p["patient_id"], true_gain=amp_scale * resistance, latent=latent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    start = pd.Timestamp("2026-01-01 00:00")
    ehr, cgm_all, meals_all, sleep_all, gt = [], [], [], [], []
    for pid in range(a.patients):
        p = make_patient(pid, rng)
        cgm, meals, sleep, g = simulate_patient(p, a.days, start, rng)
        gt.append(g)
        ehr.append(p); cgm_all.append(cgm); meals_all.append(meals); sleep_all.append(sleep)

    pd.DataFrame(ehr).to_parquet(f"{a.out}/ehr.parquet", index=False)
    pd.concat(cgm_all).to_parquet(f"{a.out}/cgm.parquet", index=False)
    pd.concat(meals_all).to_parquet(f"{a.out}/meals.parquet", index=False)
    pd.concat(sleep_all).to_parquet(f"{a.out}/sleep.parquet", index=False)
    pd.DataFrame(gt).to_parquet(f"{a.out}/ground_truth.parquet", index=False)  # evaluation only; never used by models
    print(f"Wrote {a.patients} patients x {a.days} days to {a.out}")


if __name__ == "__main__":
    main()
