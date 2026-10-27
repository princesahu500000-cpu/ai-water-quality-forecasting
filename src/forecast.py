import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from wqi import wqi_category

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

H = 7  # kitne din aage forecast karna hai

df = pd.read_csv("data/water_quality_wqi.csv", parse_dates=["date"], index_col="date")
df = df.asfreq("D")

# ---------- Features ----------
cols = ["temperature", "ph", "dissolved_oxygen", "turbidity", "conductivity", "bod", "wqi"]
feat = pd.DataFrame(index=df.index)
for c in cols:
    for lag in [0, 1, 2, 3, 7, 14]:
        feat[f"{c}_lag{lag}"] = df[c].shift(lag)
    feat[f"{c}_roll7"] = df[c].rolling(7).mean()
doy = df.index.day_of_year
feat["doy_sin"] = np.sin(2 * np.pi * doy / 365)
feat["doy_cos"] = np.cos(2 * np.pi * doy / 365)

data = feat.copy()
data["target"] = df["wqi"].shift(-H)   # H din baad ka WQI
data = data.dropna()

# ---------- Time-based split ----------
split = int(len(data) * 0.8)
train = data.iloc[: split - H]          # leakage se bachne ke liye H rows hataye
test = data.iloc[split:]
test_eval = test.iloc[::H]              # har 7 din pe ek forecast origin

X_cols = [c for c in data.columns if c != "target"]
X_train, y_train = train[X_cols], train["target"]
X_test, y_true = test_eval[X_cols], test_eval["target"]

print(f"Train: {len(train)} rows | Test forecasts: {len(test_eval)}")

preds = {}

# 1. Naive baseline
preds["Naive"] = test_eval["wqi_lag0"].values

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds["RandomForest"] = rf.predict(X_test)

# 3. XGBoost
xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                   subsample=0.9, colsample_bytree=0.9, random_state=42)
xgb.fit(X_train, y_train)
preds["XGBoost"] = xgb.predict(X_test)

# 4. SARIMAX (ARIMA + annual seasonality via sin/cos)
series = df["wqi"]
exog = pd.DataFrame({"s": np.sin(2 * np.pi * doy / 365),
                     "c": np.cos(2 * np.pi * doy / 365)}, index=df.index)
origins = list(test_eval.index)
one = pd.Timedelta(days=1)
res = SARIMAX(series.loc[:origins[0]], exog=exog.loc[:origins[0]],
              order=(2, 0, 1), trend="c").fit(disp=False)
sarimax_pred = []
for i, t in enumerate(origins):
    fut = exog.loc[t + one: t + H * one]
    sarimax_pred.append(res.forecast(steps=H, exog=fut).iloc[-1])
    if i + 1 < len(origins):
        nxt = origins[i + 1]
        res = res.append(series.loc[t + one: nxt], exog=exog.loc[t + one: nxt], refit=False)
preds["SARIMAX"] = np.array(sarimax_pred)

# ---------- Evaluation ----------
true_cat = y_true.apply(wqi_category).values
rows = []
for name, p in preds.items():
    err = y_true.values - p
    rows.append({
        "Model": name,
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE_%": np.mean(np.abs(err / y_true.values)) * 100,
        "Category_Acc_%": np.mean(np.array([wqi_category(v) for v in p]) == true_cat) * 100,
    })
results = pd.DataFrame(rows).sort_values("RMSE").round(2)
print(f"\n=== {H}-day ahead WQI forecast results ===")
print(results.to_string(index=False))
results.to_csv("report/results.csv", index=False)

# ---------- Plot ----------
plt.figure(figsize=(13, 5))
plt.plot(y_true.index + H * one, y_true.values, "k-o", markersize=3, label="Actual")
for name in ["Naive", "SARIMAX", "XGBoost"]:
    plt.plot(y_true.index + H * one, preds[name], label=name, alpha=0.8)
plt.axhline(100, color="orange", linestyle="--", linewidth=0.8)
plt.title(f"{H}-day ahead WQI forecast vs actual (test period)")
plt.ylabel("WQI")
plt.legend()
plt.tight_layout()
plt.savefig("report/figures/forecast_comparison.png", dpi=150)

# ---------- Best ML model save ----------
joblib.dump({"model": xgb, "features": X_cols, "horizon": H}, "models/xgb_wqi.pkl")
print("\nSaved: report/results.csv, forecast_comparison.png, models/xgb_wqi.pkl")
plt.show()