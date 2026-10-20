"""
GlucoTwin-IN API.

Each patient has a live twin. A per-patient time cursor simulates a streaming wearable feed:
POST /advance moves the cursor forward, completed meals are observed and the twin updates.
"""
import os
import sys
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.generate import MEALS  # noqa: E402
from models.spike_model import (HORIZON, START, STEP, _nm, _sleep_for, meal_feature_row,  # noqa: E402
                                prepare_patient_arrays)
from twin.engine import PatientTwin  # noqa: E402
from twin import foods as foodlib  # noqa: E402
from api import auth  # noqa: E402
from api.auth import User, audit, current_user, read_audit, session_user  # noqa: E402

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data/synthetic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join(ROOT, "models/artifacts"))
START_CURSOR = 10 * 288 + 7 * 12 + 30 // STEP   # day 10, 07:30
SLOT_HOURS = {"breakfast": 8.0, "lunch": 13.25, "snack": 17.0, "dinner": 20.5}

app = FastAPI(title="GlucoTwin-IN", version="1.0",
              description="Digital twin proof-of-concept for T2D post-meal glucose spikes. Synthetic data only.")

app.include_router(auth.router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api") or request.url.path in ("/", "/login"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def doctor_only(user: User = Depends(current_user)) -> User:
    if user.role != "doctor":
        raise HTTPException(403, "This view is for doctors.")
    return user


def patient_access(pid: int, user: User = Depends(current_user)) -> User:
    if user.role == "patient" and user.patient_id != pid:
        raise HTTPException(403, "You can only open your own record.")
    return user


# ---------------------------------------------------------------- data / models (loaded once)
@lru_cache(maxsize=1)
def store():
    ehr = pd.read_parquet(f"{DATA_DIR}/ehr.parquet").set_index("patient_id")
    cgm = pd.read_parquet(f"{DATA_DIR}/cgm.parquet")
    meals = pd.read_parquet(f"{DATA_DIR}/meals.parquet")
    sleep = pd.read_parquet(f"{DATA_DIR}/sleep.parquet")
    return dict(
        ehr=ehr,
        cgm={pid: g.sort_values("timestamp") for pid, g in cgm.groupby("patient_id")},
        meals={pid: g.sort_values("timestamp").reset_index(drop=True) for pid, g in meals.groupby("patient_id")},
        sleep={pid: g for pid, g in sleep.groupby("patient_id")},
        cls=lgb.Booster(model_file=f"{ART_DIR}/spike_cls.txt"),
        reg=lgb.Booster(model_file=f"{ART_DIR}/spike_reg.txt"),
        meta=__import__("json").load(open(f"{ART_DIR}/spike_meta.json")),
        arrays={}, raw={}, sessions={},
    )


class Session:
    def __init__(self, pid):
        S = store()
        self.pid = pid
        self.ehr = S["ehr"].loc[pid].to_dict()
        self.raw, self.g, self.hr, self.steps = prepare_patient_arrays(S["cgm"][pid])
        self.meals = S["meals"][pid]
        self.sleep = S["sleep"][pid]
        self.twin = PatientTwin(self.ehr)
        self.cursor = START_CURSOR
        self.mi = 0
        self.outcomes = []        # observed meal outcomes (for the dashboard log)
        self._catch_up()

    def _meal_idx(self, ts):
        return int((ts - START).total_seconds() // 60 // STEP)

    def _catch_up(self):
        """Observe every meal whose 2 h outcome is complete at the current cursor."""
        new = []
        while self.mi < len(self.meals):
            r = self.meals.iloc[self.mi]
            i = self._meal_idx(r.timestamp)
            if i + HORIZON > self.cursor:
                break
            self.mi += 1
            w = self.raw[i:i + HORIZON + 1]
            pre = _nm(self.g[max(i - 2, 0):i + 1])
            if i < 72 or not np.isfinite(w).any() or not np.isfinite(pre):
                continue
            rise = float(np.nanmax(w) - pre)
            walked = bool(np.nansum(self.steps[i + 2:i + 9]) >= 150)
            sh, _ = _sleep_for(self.sleep, r.timestamp)
            predicted = self.twin.predict_rise(r.glycemic_load, walked, sh)
            self.twin.observe_meal(r.glycemic_load, rise, walked, sh)
            o = dict(time=r.timestamp.isoformat(), meal=r.meal, slot=r.slot, gl=float(r.glycemic_load),
                     predicted_rise=round(predicted, 1), observed_rise=round(rise, 1), walked=walked,
                     spike=bool(rise >= 60))
            self.outcomes.append(o)
            new.append(o)
        return new

    def now(self):
        return START + pd.Timedelta(minutes=STEP * self.cursor)

    def current_glucose(self):
        v = self.g[self.cursor]
        return float(v) if np.isfinite(v) else float(np.nanmean(self.g[self.cursor - 12:self.cursor + 1]))

    def last_sleep(self):
        return _sleep_for(self.sleep, self.now())


def session(pid: int) -> Session:
    S = store()
    if pid not in S["ehr"].index:
        raise HTTPException(404, f"patient {pid} not found")
    if pid not in S["sessions"]:
        S["sessions"][pid] = Session(pid)
    return S["sessions"][pid]


def upcoming_slot(hour: float) -> str:
    for slot, h in SLOT_HOURS.items():
        if h >= hour - 1.0:
            return slot
    return "breakfast"


def typical_meal(se: Session, slot: str) -> str:
    m = se.meals[(se.meals.slot == slot) & (se.meals.timestamp < se.now())]
    return m.meal.mode().iloc[0] if len(m) else MEALS[slot][0][0]


def ml_predict(se: Session, slot: str, carbs: float, gi: float, portion: float):
    i = se.cursor
    prev = None
    hist = se.meals[se.meals.timestamp <= se.now()]
    if len(hist):
        pm = hist.iloc[-1]
        prev = dict(mins_ago=(se.now() - pm.timestamp).total_seconds() / 60, gl=float(pm.glycemic_load))
    sh, dp = se.last_sleep()
    meal = dict(slot=slot, carbs_g=carbs * portion, gi=gi, glycemic_load=carbs * portion * gi / 100)
    row = meal_feature_row(se.g, se.hr, se.steps, i, meal, prev, sh, dp, se.ehr)
    X = pd.DataFrame([row])[store()["meta"]["features"]]
    return float(store()["cls"].predict(X)[0]), float(store()["reg"].predict(X)[0])


# ---------------------------------------------------------------- schemas
class Scenario(BaseModel):
    label: str = "Scenario"
    meal: str | None = None          # a library meal name (compatibility)
    foods: str | None = None         # free text, e.g. "2 idli, sambar, coconut chutney"
    portion: float = Field(1.0, ge=0.3, le=2.5)
    walk: bool = False
    sleep_hours: float | None = Field(None, ge=3, le=10)


class FoodText(BaseModel):
    text: str = Field("", max_length=300)


class WhatIf(BaseModel):
    scenarios: list[Scenario]


class Advance(BaseModel):
    hours: float = Field(3.0, gt=0, le=24)


class IngestReading(BaseModel):
    patient_id: int = Field(..., ge=0)
    glucose: float = Field(..., ge=20, le=600)
    heart_rate: float = Field(..., ge=25, le=240)
    steps: int = Field(..., ge=0, le=10000)
    sleep_hours: float | None = Field(None, ge=0, le=24)
    meal: str | None = Field(None, max_length=300)


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"status": "ok", "patients": int(len(store()["ehr"]))}


@app.get("/api/meals")
def meal_library(user: User = Depends(current_user)):
    return {slot: [dict(name=n, carbs_g=c, gi=g, region=r) for n, c, g, r in items] for slot, items in MEALS.items()}


@app.get("/api/foods/search")
def foods_search(q: str = "", limit: int = 6, user: User = Depends(current_user)):
    return foodlib.search(q[:60], min(limit, 10))


@app.post("/api/foods/parse")
def foods_parse(body: FoodText, user: User = Depends(current_user)):
    return foodlib.parse_meal(body.text)


@app.post("/api/foods/swap")
def foods_swap(body: FoodText, user: User = Depends(current_user)):
    return foodlib.suggest_swap(body.text)


@app.get("/api/model-card")
def model_card(user: User = Depends(doctor_only)):
    import json
    out = dict(spike=store()["meta"]["metrics"])
    for f in ("ablation", "twin_eval"):
        p = f"{ART_DIR}/{f}.json"
        if os.path.exists(p):
            out[f] = json.load(open(p))
    return out


def cohort_row(pid: int, e) -> dict:
    """Light-weight triage numbers at the patient's current time (no twin replay needed)."""
    S = store()
    raw = S["raw"].setdefault(pid, S["cgm"][pid]["glucose"].to_numpy(float))
    i = S["sessions"][pid].cursor if pid in S["sessions"] else START_CURSOR
    w = raw[max(i - 287, 0):i + 1]
    ok = w[np.isfinite(w)]
    tir = float(np.mean((ok >= 70) & (ok <= 180)) * 100) if len(ok) else None
    recent = raw[max(i - 12, 0):i + 1]
    recent = recent[np.isfinite(recent)]
    now = float(recent[-1]) if len(recent) else None
    t_now = START + pd.Timedelta(minutes=STEP * i)
    m = S["meals"].get(pid)
    spikes = meals_24h = 0
    if m is not None:
        for ts in m.timestamp[(m.timestamp > t_now - pd.Timedelta(hours=24)) & (m.timestamp <= t_now - pd.Timedelta(hours=2))]:
            k = int((ts - START).total_seconds() // 60 // STEP)
            pre = raw[max(k - 2, 0):k + 1]; post = raw[k:k + HORIZON + 1]
            if np.isfinite(pre).any() and np.isfinite(post).any():
                meals_24h += 1
                spikes += int(np.nanmax(post) - np.nanmean(pre) >= 60)
    return dict(id=int(pid), age=int(e.age), sex=e.sex, hba1c=float(e.hba1c), bmi=float(e.bmi),
                glucose_now=None if now is None else round(now), tir_24h=None if tir is None else round(tir),
                spikes_24h=spikes, meals_24h=meals_24h, attention=False,
                score=round((100 - (tir or 0)) + 12 * spikes + max(0.0, (now or 0) - 180) / 4, 1))


@app.get("/api/patients")
def patients(limit: int = 300, user: User = Depends(doctor_only)):
    S = store()
    rows = [cohort_row(int(pid), e) for pid, e in S["ehr"].head(limit).iterrows()]
    cut = float(np.percentile([r["score"] for r in rows], 85)) if rows else 0.0     # worklist: top 15% by risk score today
    for r in rows:
        r["attention"] = bool(r["score"] >= cut and ((r["tir_24h"] or 100) < 50 or r["spikes_24h"] >= 2 or (r["glucose_now"] or 0) >= 250))
    return rows


@app.get("/api/patients/{pid}")
def patient(pid: int, hours: int = 36, user: User = Depends(patient_access)):
    se = session(pid)
    audit(user, "view_patient", pid)
    e = {k: (v.item() if hasattr(v, "item") else v) for k, v in se.ehr.items()}
    i0 = max(se.cursor - hours * 12, 0)
    ts = [(START + pd.Timedelta(minutes=STEP * k)).isoformat() for k in range(i0, se.cursor + 1)]
    gl = [None if not np.isfinite(v) else round(float(v), 1) for v in se.raw[i0:se.cursor + 1]]
    hist = se.meals[(se.meals.timestamp >= START + pd.Timedelta(minutes=STEP * i0)) & (se.meals.timestamp <= se.now())]
    meals = [dict(time=r.timestamp.isoformat(), meal=r.meal, slot=r.slot, gl=float(r.glycemic_load)) for _, r in hist.iterrows()]
    window = se.raw[max(se.cursor - 24 * 12, 0):se.cursor + 1]
    tir = float(np.nanmean((window >= 70) & (window <= 180)) * 100)
    slot = upcoming_slot((se.cursor * STEP % 1440) / 60)
    sh, dp = se.last_sleep()
    lo, hi = se.twin.ci()
    return dict(
        ehr=e, now=se.now().isoformat(), glucose_now=round(se.current_glucose(), 1), tir_24h=round(tir),
        last_sleep_hours=None if sh is None or not np.isfinite(sh) else round(sh, 1),
        timeline=dict(t=ts, glucose=gl), meals=meals,
        twin=dict(gain=round(se.twin.gain, 3), ci_low=round(lo, 3), ci_high=round(hi, 3), meals_observed=se.twin.n_obs,
                  history=se.twin.history[-60:]),
        recent_outcomes=se.outcomes[-6:][::-1],
        suggested=dict(slot=slot, meal=typical_meal(se, slot), text=typical_meal(se, slot).replace("_", " ")),
    )


@app.post("/api/patients/{pid}/whatif")
def whatif(pid: int, body: WhatIf, user: User = Depends(patient_access)):
    se = session(pid)
    audit(user, "what_if", pid)
    sh_default, _ = se.last_sleep()
    g0 = se.current_glucose()
    out = []
    slot = upcoming_slot((se.cursor * STEP % 1440) / 60)
    for sc in body.scenarios:
        text = (sc.foods or "").strip() or (sc.meal or "").replace("_", " ").strip()
        if not text:
            raise HTTPException(422, "Type what the patient is eating.")
        pm = foodlib.parse_meal(text)
        if pm["carbs_g"] <= 0:
            raise HTTPException(422, "Couldn't find: " + (", ".join(pm["unmatched"]) or text) + ". Try a common Indian food name, or add carbs like 'pizza 40 g'.")
        sh = sc.sleep_hours if sc.sleep_hours is not None else (sh_default if np.isfinite(sh_default) else None)
        sim = se.twin.simulate(g0, pm["carbs_g"], pm["gi"], sc.portion, sc.walk, sh)
        p_ml, rise_ml = ml_predict(se, slot, pm["carbs_g"], pm["gi"], sc.portion)
        out.append(dict(label=sc.label, foods=text, parsed=pm, **sim, ml_p_spike=round(p_ml, 3), ml_rise=round(rise_ml, 1),
                        p_spike_ensemble=round((p_ml + sim["p_spike"]) / 2, 3)))
    return dict(start_glucose=round(g0, 1), scenarios=out,
                note="Twin curve responds to walk/sleep/portion; the ML model uses the meal and history only (walks are future actions).")


@app.post("/api/patients/{pid}/advance")
def advance(pid: int, body: Advance, user: User = Depends(patient_access)):
    se = session(pid)
    audit(user, "stream_data", pid)
    limit = len(se.raw) - 1 - HORIZON
    se.cursor = min(se.cursor + int(body.hours * 12), limit)
    new = se._catch_up()
    return dict(now=se.now().isoformat(), cursor=se.cursor, reached_end=se.cursor >= limit, newly_observed=new,
                twin=dict(gain=round(se.twin.gain, 3), meals_observed=se.twin.n_obs))


@app.post("/api/ingest")
def ingest(body: IngestReading, user: User = Depends(doctor_only)):
    """Append one synthetic wearable reading without changing the historical cohort."""
    se = session(body.patient_id)
    if not hasattr(se, "live_readings"):
        se.live_readings = []
    previous = se.live_readings[-1] if se.live_readings else None
    reading = dict(glucose=float(body.glucose), heart_rate=float(body.heart_rate),
                   steps=int(body.steps), sleep_hours=body.sleep_hours, meal=body.meal)
    se.live_readings.append(reading)
    update = None
    if body.meal:
        parsed = foodlib.parse_meal(body.meal)
        if parsed["carbs_g"] <= 0:
            se.live_readings.pop()
            raise HTTPException(422, "Couldn't find: " + (", ".join(parsed["unmatched"]) or body.meal) + ".")
        baseline = float(previous["glucose"]) if previous else se.current_glucose()
        # A single reading cannot establish a negative rise; use the smallest
        # admissible observation so the optional meal still enters the filter.
        observed_rise = max(float(body.glucose) - baseline, 2.01)
        sleep_h = body.sleep_hours if body.sleep_hours is not None else se.last_sleep()[0]
        walked = body.steps >= 100
        se.twin.observe_meal(parsed["glycemic_load"], observed_rise, walked, sleep_h)
        update = dict(meal=body.meal, glycemic_load=round(parsed["glycemic_load"], 1),
                      observed_rise=round(observed_rise, 1), meals_observed=se.twin.n_obs)
    audit(user, "ingest_reading", body.patient_id)
    return dict(patient_id=body.patient_id, reading=reading, twin=dict(
        gain=round(se.twin.gain, 3), meals_observed=se.twin.n_obs), update=update)


@app.post("/api/patients/{pid}/reset")
def reset(pid: int, user: User = Depends(patient_access)):
    audit(user, "reset_twin", pid)
    store()["sessions"].pop(pid, None)
    return {"ok": True}


@app.get("/api/audit")
def audit_log(user: User = Depends(doctor_only)):
    return read_audit(30)


# ---------------------------------------------------------------- pages
STATIC = os.path.join(ROOT, "dashboard/static")
PAGES = os.path.join(ROOT, "dashboard/pages")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
auth.patient_exists = lambda pid: pid in store()["ehr"].index


@app.get("/")
def index(request: Request):
    if not session_user(request):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(os.path.join(PAGES, "index.html"))


@app.get("/login")
def login_page(request: Request):
    if session_user(request):
        return RedirectResponse("/", status_code=302)
    return FileResponse(os.path.join(PAGES, "login.html"))
