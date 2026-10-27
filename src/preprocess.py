import pandas as pd
import numpy as np

df = pd.read_csv("data/water_quality_raw.csv", parse_dates=["date"])
df = df.sort_values("date").set_index("date")

# 1. Physically impossible values ko NaN karo
valid_range = {
    "temperature": (0, 45),
    "ph": (4, 10),
    "dissolved_oxygen": (0, 20),
    "turbidity": (0, 500),
    "conductivity": (0, 2000),
    "bod": (0, 30),
}
for col, (lo, hi) in valid_range.items():
    bad = (df[col] < lo) | (df[col] > hi)
    print(f"{col}: {bad.sum()} invalid values removed")
    df.loc[bad, col] = np.nan

# 2. Daily frequency + missing values fill
num_cols = list(valid_range.keys())
df = df[num_cols].resample("D").mean()
print("\nMissing before fill:\n", df.isna().sum())
df = df.interpolate(method="time").bfill().ffill()

df.to_csv("data/water_quality_clean.csv")
print("\nSaved clean data:", df.shape)