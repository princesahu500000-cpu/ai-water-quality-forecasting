"""Run the whole project from start to finish (no plot windows, no manual steps).

Usage:  python run_all.py
Order:  synthetic part (generate, clean, EDA, WQI, forecast) -> real part -> done.
Afterwards start the demo app with:  streamlit run src/app.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Generate synthetic data", "src/generate_data.py"),
    ("Clean synthetic data", "src/preprocess.py"),
    ("Exploratory graphs", "src/eda.py"),
    ("WQI (synthetic)", "src/wqi.py"),
    ("7-day forecasting (synthetic)", "src/forecast.py"),
    ("Real data: WQI and next-year forecasting", "src/real_data_forecast.py"),
]

env = dict(os.environ, MPLBACKEND="Agg")      # save figures, never open windows
real_csv = ROOT / "data" / "water_dataX.csv"
if not real_csv.exists():
    sys.exit("Missing data/water_dataX.csv. Download it from "
             "https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data "
             "and place it in the data folder (see README.md).")

for number, (label, script) in enumerate(STEPS, start=1):
    print(f"\n[{number}/{len(STEPS)}] {label}  ({script})", flush=True)
    subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True)

print("\nAll steps finished. Start the demo with:  streamlit run src/app.py")