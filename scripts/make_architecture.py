"""Draws docs/architecture.pdf and docs/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, TEAL, BLUE, AMBER, PAPER = "#10272b", "#0e7c73", "#3b5bdb", "#c98a12", "#f8faf9"
fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 130); ax.set_ylim(0, 72); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, title, lines, color, fc=PAPER):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.6, "\n".join(lines), ha="center", va="top", fontsize=8.3, color="#33474b", linespacing=1.45)

def arrow(p, q, color=INK, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.6, label, ha="center", fontsize=7.8, color=color, style="italic")

ax.text(65, 69.5, "GlucoTwin India: architecture", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(65, 66.2, "Static record + live wearable stream  ->  personalized twin  ->  spike forecast and what-if", ha="center", fontsize=9.5, color="#5b7074")

# data layer
ax.text(2, 61, "1  Data (synthetic only)", fontsize=9.5, fontweight="bold", color=TEAL)
box(2, 40, 30, 18, "Static stream: EHR", ["age, sex, BMI, duration", "HbA1c, eGFR, LDL, TG", "TCF7L2 genotype, adherence", "diet pattern, region, activity"], TEAL)
box(2, 16, 30, 21, "Dynamic stream: wearables", ["CGM glucose (5 min)", "heart rate, steps, sleep", "Indian meal log: carbs, GI,", "glycemic load", "sensor gaps, unlogged meals"], BLUE)
ax.text(17, 11.2, "Coupled through insulin resistance:\nEHR traits scale every meal response", ha="center", fontsize=8, color=INK, style="italic")

# feature + models
ax.text(40, 61, "2  Fusion and models", fontsize=9.5, fontweight="bold", color=TEAL)
box(40, 40, 30, 18, "Feature builder", ["meal-time snapshot (no leakage):", "CGM level + trend, HR, steps,", "last night's sleep, previous meal,", "static EHR features"], INK)
box(40, 16, 30, 21, "A  Spike model (LightGBM)", ["P(rise >= 60 mg/dL within 2 h)", "+ expected rise (regressor)", "patient-level train/val/test split", "baselines + fusion ablation"], AMBER)
box(75, 40, 28, 18, "B  Digital twin", ["state: patient response gain", "HbA1c prior -> Kalman update", "after every observed meal", "95% credible band"], TEAL)
box(75, 16, 28, 21, "What-if simulator", ["meal swap (rice -> millet)", "portion size, post-meal walk,", "sleep hours", "forecast curve + uncertainty band"], BLUE)

# service
ax.text(108, 61, "3  Service and UI", fontsize=9.5, fontweight="bold", color=TEAL)
box(108, 40, 21, 18, "FastAPI", ["/patients  /advance", "/whatif  /model-card", "streaming cursor"], INK)
box(108, 16, 21, 21, "Doctor dashboard", ["virtual patient view", "plan A vs plan B", "twin state + recent meals", "evidence panel"], TEAL)

arrow((32, 49), (40, 49)); arrow((32, 33), (40, 43), rad=-0.2)
arrow((55, 40), (55, 37)); arrow((70, 49), (75, 49), label="")
arrow((70, 30), (75, 30), color=AMBER)
arrow((89, 40), (89, 37), color=TEAL)
arrow((103, 49), (108, 49)); arrow((103, 27), (108, 27)); arrow((118.5, 40), (118.5, 37), color=TEAL)
ax.text(89, 38.2, "", fontsize=7)
ax.text(65, 6.5, "Loop: each logged meal is observed 2 h later  ->  twin gain updates  ->  next forecast is sharper", ha="center", fontsize=9, color=TEAL, fontweight="bold")
ax.text(65, 2.5, "Privacy: no real patient data is used. Docker build generates the cohort (seed 42), trains and evaluates; the container serves API and dashboard on :8000", ha="center", fontsize=8, color="#5b7074")
fig.savefig("docs/architecture.pdf", bbox_inches="tight"); fig.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
