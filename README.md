---
title: AI for Exoplanets KOI
emoji: 🪐
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: NASA Kepler KOI research demo + CatBoost planet vs FP
---

# AI for Exoplanets Discovery (Kepler KOI)

Interactive demo by **Kyrylo Yeromenko**: exploratory charts on NASA Kepler KOI data and a CatBoost classifier (planet vs false positive) first designed in Orange, then reproduced for this Space.

## Tabs

- **Research** — EDA charts and conclusions  
- **Model** — metrics, confusion matrix, feature importance, Orange workflow  
- **Predict** — manual / random / batch scoring  

## Local run

```bash
cd challenge   # or Space repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7865

## Data

NASA Exoplanet Archive — KOI cumulative.  
Training labels: `CONFIRMED` vs `FALSE POSITIVE` only (`CANDIDATE` for demo prediction).
