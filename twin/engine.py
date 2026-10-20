"""
Personalized hybrid digital twin of a T2D patient.

Hybrid = mechanistic response model (parameters per patient) + online Bayesian state update.

State (per patient):  m = log of the patient's insulin-response gain G (mg/dL peak rise per unit
glycemic load),  v = variance of that estimate.  Prior comes from HbA1c only; the remaining
patient-specific physiology (BMI, genotype, adherence, ...) is LEARNED from the patient's own
logged meals + CGM, so the twin sharpens as data streams in (scalar Kalman filter in log-space).

What-if simulator: predicted glucose curve for any hypothetical meal / walk / sleep / portion.
Context effects (post-meal walk, short sleep) use literature-style priors, not the generator's truth.

All parameters are assumptions for a proof of concept, not clinically validated values.
"""
from dataclasses import dataclass, field
import numpy as np

STEP = 5
WALK_FACTOR = 0.75        # prior: ~25% lower excursion with a 10-20 min post-meal walk
SLEEP_SLOPE = 0.08        # prior: +8% response per hour below 7 h sleep
OBS_SD = 0.55             # log-space observation noise per meal (overlap, sensor noise, unlogged events)
OBS_BIAS = 0.10           # peak-of-noisy-signal upward bias; calibrated on a separate cohort (patients 0-119)
PROCESS_SD = 0.01         # per-meal drift of true gain (synthetic physiology is stationary; real patients need larger)
SPIKE_RISE = 60.0
HIGH = 180.0


def meal_curve(gi: float, n_steps: int = 72) -> np.ndarray:
    tp = 90 - 0.5 * gi
    x = (np.arange(n_steps) * STEP + 1e-6) / tp
    return x ** 2 * np.exp(2 * (1 - x))


def context_factor(walked: bool, sleep_h: float | None) -> float:
    f = WALK_FACTOR if walked else 1.0
    if sleep_h is not None and np.isfinite(sleep_h):
        f *= 1 + SLEEP_SLOPE * max(0.0, 7.0 - sleep_h)
    return f


def prior_gain(hba1c: float) -> tuple[float, float]:
    """HbA1c-only prior for (log gain, variance). Deliberately coarse."""
    return float(np.log(1.5) + 0.08 * (hba1c - 7.7)), 0.50 ** 2


@dataclass
class PatientTwin:
    ehr: dict
    m: float = 0.0
    v: float = 0.0
    n_obs: int = 0
    history: list = field(default_factory=list)   # (n_obs, gain, ci_lo, ci_hi) for dashboard

    def __post_init__(self):
        self.m, self.v = prior_gain(self.ehr["hba1c"])
        self._log()

    @property
    def gain(self) -> float:
        return float(np.exp(self.m))

    def ci(self, z=1.96):
        s = np.sqrt(self.v)
        return float(np.exp(self.m - z * s)), float(np.exp(self.m + z * s))

    def _log(self):
        lo, hi = self.ci()
        self.history.append(dict(n=self.n_obs, gain=round(self.gain, 3), lo=round(lo, 3), hi=round(hi, 3)))

    # ---- live update: call when a meal's 2 h outcome has been observed ----
    def observe_meal(self, glycemic_load: float, observed_rise: float, walked: bool, sleep_h: float | None):
        if glycemic_load < 3 or observed_rise <= 2:
            return
        adj = glycemic_load * context_factor(walked, sleep_h)
        y = np.log(max(observed_rise, 2.0) / adj) - OBS_BIAS   # observed log gain, bias-corrected
        self.v += PROCESS_SD ** 2                          # predict
        k = self.v / (self.v + OBS_SD ** 2)                # Kalman gain
        self.m += k * (y - self.m)
        self.v *= (1 - k)
        self.n_obs += 1
        self._log()

    # ---- prediction / simulation ----
    def predict_rise(self, glycemic_load, walked=False, sleep_h=None) -> float:
        return self.gain * glycemic_load * context_factor(walked, sleep_h)

    def simulate(self, g_start: float, carbs_g: float, gi: float, portion: float = 1.0,
                 walked: bool = False, sleep_h: float | None = None, minutes: int = 240) -> dict:
        gl = carbs_g * portion * gi / 100.0
        amp = self.predict_rise(gl, walked, sleep_h)
        lo_g, hi_g = np.exp(self.m - 1.28 * np.sqrt(self.v + OBS_SD ** 2)), np.exp(self.m + 1.28 * np.sqrt(self.v + OBS_SD ** 2))
        scale = amp / self.gain
        n = minutes // STEP
        c = meal_curve(gi, n)
        t = np.arange(n) * STEP
        curve = g_start + amp * c
        lo, hi = g_start + lo_g * scale * c, g_start + hi_g * scale * c
        peak_i = int(np.argmax(curve))
        above = np.clip(curve - HIGH, 0, None)
        rise = float(curve.max() - g_start)
        # P(spike): rise ~ lognormal around amp; spike if rise >= SPIKE_RISE
        sd = np.sqrt(self.v + OBS_SD ** 2)
        from math import erf, sqrt, log
        z = (log(SPIKE_RISE) - log(max(amp, 1e-6))) / sd
        p_spike = 0.5 * (1 - erf(z / sqrt(2)))
        return dict(minutes=t.tolist(), glucose=curve.round(1).tolist(), lo=lo.round(1).tolist(), hi=hi.round(1).tolist(),
                    peak=round(float(curve.max()), 1), peak_minute=int(t[peak_i]), rise=round(rise, 1),
                    minutes_above_180=int((curve > HIGH).sum() * STEP), auc_above_180=round(float(above.sum() * STEP), 0),
                    p_spike=round(float(p_spike), 3), glycemic_load=round(gl, 1))
