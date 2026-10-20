#!/usr/bin/env sh
# Regenerate data, retrain, re-evaluate. Deterministic (seed 42).
set -e
python data/generate.py --patients 300 --days 14 --seed 42
python data/sanity_check.py
python models/spike_model.py
python models/ablation.py
python twin/evaluate.py
