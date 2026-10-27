import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
n = len(dates)
doy = dates.day_of_year.values


def ar_noise(n, phi=0.8, sigma=1.0):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


monsoon = np.exp(-((doy - 225) / 35) ** 2)

temperature = 24 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + ar_noise(n, 0.7, 0.6)
turbidity = 20 + 130 * monsoon + 6 * np.abs(ar_noise(n, 0.85, 1.0))
dissolved_oxygen = 14.6 - 0.25 * temperature - 0.012 * turbidity + ar_noise(n, 0.7, 0.25)
ph = 7.7 - 0.5 * monsoon + 0.15 * np.sin(2 * np.pi * doy / 365) + ar_noise(n, 0.8, 0.05)
conductivity = 450 - 220 * monsoon + 8 * ar_noise(n, 0.85, 1.0)
bod = 2.0 + 0.06 * temperature + 3.0 * monsoon + 0.3 * ar_noise(n, 0.8, 0.5)

df = pd.DataFrame({
    "date": dates,
    "station": "Station_A",
    "temperature": temperature.round(2),
    "ph": ph.round(2),
    "dissolved_oxygen": dissolved_oxygen.round(2),
    "turbidity": turbidity.round(2),
    "conductivity": conductivity.round(1),
    "bod": bod.round(2),
})

cols = ["temperature", "ph", "dissolved_oxygen", "turbidity", "conductivity", "bod"]
for c in cols:
    idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[idx, c] = np.nan

out_idx = rng.choice(n, size=15, replace=False)
df.loc[out_idx[:5], "ph"] = 14.9
df.loc[out_idx[5:10], "turbidity"] = 999
df.loc[out_idx[10:], "dissolved_oxygen"] = -5

df.to_csv("data/water_quality_raw.csv", index=False)
print("Saved:", df.shape)
print(df.head())