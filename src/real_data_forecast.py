import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.makedirs("report/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================================
# 1. Load + clean
# =====================================================
raw = pd.read_csv("data/water_dataX.csv", encoding="ISO-8859-1")
raw = raw.rename(columns={
    "STATION CODE": "station", "LOCATIONS": "location", "STATE": "state",
    "Temp": "temperature", "D.O. (mg/l)": "do", "PH": "ph",
    "CONDUCTIVITY (µmhos/cm)": "ec", "B.O.D. (mg/l)": "bod",
    "NITRATENAN N+ NITRITENANN (mg/l)": "nitrate",
    "FECAL COLIFORM (MPN/100ml)": "fecal_coliform",
    "TOTAL COLIFORM (MPN/100ml)Mean": "tc", "year": "year",
})
PARAMS = ["temperature", "do", "ph", "ec", "bod", "nitrate", "fecal_coliform", "tc"]
for c in PARAMS + ["year"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")   # "NAN" etc. -> NaN (not 0!)
# Treat station/location/state entries containing the text "NAN" as missing (otherwise, all "NAN" values become one fake station)
for c in ["station", "location", "state"]:
    raw[c] = raw[c].astype(str).str.strip().replace({"NAN": np.nan, "nan": np.nan, "": np.nan})

# In several rows, the STATE column contains location text instead of the actual state.
# (e.g., "ACHENKOIL AT CHENNITHULA, KERALA"). Separate the state and location
KNOWN_STATES = {
    "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHANDIGARH", "CHHATTISGARH", "DELHI",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA",
    "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND",
    "ODISHA", "PUDUCHERRY", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TELANGANA", "TRIPURA",
    "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "DAMAN & DIU", "DAMAN, DIU, DADRA NAGAR HAVELI",
}
STATE_ALIAS = {"TAMILNADU": "TAMIL NADU", "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY",
               "AP": "ANDHRA PRADESH", "A.P": "ANDHRA PRADESH", "H.P": "HIMACHAL PRADESH",
               "DAMAN": "DAMAN & DIU"}
def _canon(t):
    t = re.sub(r"\s+", " ", str(t).strip().upper()).rstrip(". ")
    return STATE_ALIAS.get(t, t)

def split_state(val):
    """returns (clean_state, location_text_or_None)"""
    if pd.isna(val):
        return np.nan, None
    t = _canon(val)
    if t in KNOWN_STATES:
        return t, None
    last = _canon(str(val).split(",")[-1])        # "... , KERALA" -> "KERALA"
    return (last if last in KNOWN_STATES else np.nan), str(val).strip()

pairs = raw["state"].apply(split_state)
raw["state"] = pairs.apply(lambda p: p[0])
loc_from_state = pairs.apply(lambda p: p[1])
raw["location"] = raw["location"].fillna(loc_from_state)

def state_from_location(text):
    """Extract the state from the end of a location: "..., KERALA" or "... (HARYANA)" """
    if pd.isna(text):
        return np.nan
    t = str(text).strip()
    last = _canon(t.split(",")[-1])
    if last in KNOWN_STATES:
        return last
    m = re.search(r"\(([^()]+)\)\s*\.?\s*$", t)           # "(HARYANA)"
    if m and _canon(m.group(1)) in KNOWN_STATES:
        return _canon(m.group(1))
    return np.nan

raw["state"] = raw["state"].fillna(raw["location"].apply(state_from_location))

raw["station"] = raw["station"].fillna(raw["location"])      #  If the code is missing, use the location name.
raw = raw.dropna(subset=["year", "station"])                  #  Remove entries that cannot be identified
raw["year"] = raw["year"].astype(int)

# Remove physically impossible values.
limits = {"temperature": (0, 45), "do": (0, 20), "ph": (4, 11), "ec": (0, 20000),
          "bod": (0, 500), "nitrate": (0, 500)}
for c, (lo, hi) in limits.items():
    raw.loc[(raw[c] < lo) | (raw[c] > hi), c] = np.nan

# One station + one year = one row (average duplicates).
df = (raw.groupby(["station", "year"], as_index=False)
         .agg({**{c: "mean" for c in PARAMS}, "location": "first", "state": "first"}))
# Ek station ke kisi saal me state mila ho to baaki saalon me bhi wahi lagao
df["state"] = df.groupby("station")["state"].transform(lambda x: x.ffill().bfill())
print(f"Raw rows: {len(raw)} | Station-year rows: {len(df)} | Stations: {df['station'].nunique()}")

# =====================================================
# 2. WQI (0–100, higher = better water quality), banded sub-index approach.
# Missing parameters are not treated as 0; available weights are renormalized.
# =====================================================
def band(x, cuts, scores=(100, 80, 60, 40, 0)):
    """cuts: ascending upper limits (low value = good). x NaN -> NaN"""
    if pd.isna(x):
        return np.nan
    for cut, s in zip(cuts, scores):
        if x <= cut:
            return s
    return scores[-1]

def band_high(x, cuts):
    """High value = good (DO). cuts descending lower limits."""
    if pd.isna(x):
        return np.nan
    for cut, s in zip(cuts, (100, 80, 60, 40)):
        if x >= cut:
            return s
    return 0

def ph_score(x):
    if pd.isna(x):
        return np.nan
    if 6.5 <= x <= 8.5: return 100
    if 6.0 <= x < 6.5 or 8.5 < x <= 9.0: return 80
    if 5.5 <= x < 6.0 or 9.0 < x <= 9.5: return 60
    if 5.0 <= x < 5.5 or 9.5 < x <= 10.0: return 40
    return 0

sub = pd.DataFrame(index=df.index)
sub["ph"] = df["ph"].apply(ph_score)
sub["do"] = df["do"].apply(lambda v: band_high(v, (6, 5, 4, 3)))
sub["bod"] = df["bod"].apply(lambda v: band(v, (3, 6, 10, 30)))
sub["tc"] = df["tc"].apply(lambda v: band(v, (5, 50, 500, 10000)))
sub["ec"] = df["ec"].apply(lambda v: band(v, (750, 1500, 2250, 3000)))
sub["nitrate"] = df["nitrate"].apply(lambda v: band(v, (20, 50, 100, 200)))

W = {"ph": 0.165, "do": 0.281, "bod": 0.234, "tc": 0.281, "ec": 0.009, "nitrate": 0.028}
w = pd.Series(W)
avail = sub.notna()
df["wqi"] = (sub.fillna(0) * w).sum(axis=1) / (avail * w).sum(axis=1)
df.loc[avail.sum(axis=1) < 4, "wqi"] = np.nan       #  At least 4 parameters are required.
df = df.dropna(subset=["wqi"]).reset_index(drop=True)

def wqi_class(v):
    if v >= 90: return "Excellent"
    if v >= 70: return "Good"
    if v >= 50: return "Medium"
    if v >= 25: return "Bad"
    return "Very Bad"

df["category"] = df["wqi"].apply(wqi_class)
df.to_csv("data/real_station_year_wqi.csv", index=False)
print(f"Rows with valid WQI: {len(df)}")
print(df["category"].value_counts().to_string())

# Yearly trend (for report)
trend = df.groupby("year")["wqi"].agg(["mean", "count"])
plt.figure(figsize=(9, 4))
plt.plot(trend.index, trend["mean"], marker="o")
plt.title("Mean WQI of monitored Indian river stations by year (higher = better)")
plt.xlabel("Year"); plt.ylabel("Mean WQI"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("report/figures/real_yearly_trend.png", dpi=150)
plt.close()

# =====================================================
# 3.  Panel dataset: Features from year t -> WQI for year t+1
# =====================================================
d = df.copy()
d["log_tc"] = np.log1p(d["tc"])
d["log_fecal"] = np.log1p(d["fecal_coliform"])

nxt = d[["station", "year", "wqi"]].copy()
nxt["year"] -= 1
nxt = nxt.rename(columns={"wqi": "target"})
prev = d[["station", "year", "wqi"]].copy()
prev["year"] += 1
prev = prev.rename(columns={"wqi": "wqi_prev"})

data = (d.merge(nxt, on=["station", "year"], how="inner")      # “t+1” is necessary.
          .merge(prev, on=["station", "year"], how="left"))     # t-1 optional
data["wqi_change"] = data["wqi"] - data["wqi_prev"]

FEATS = ["temperature", "do", "ph", "ec", "bod", "nitrate", "log_tc", "log_fecal",
         "wqi", "wqi_prev", "wqi_change"]

# Time-based split: train = origin <= 2011, test = origin 2012, 2013 (target 2013, 2014)
train = data[data["year"] <= 2011]
test = data[data["year"] >= 2012]
print(f"\nForecast pairs -> Train: {len(train)} (origin 2003-2011) | Test: {len(test)} (origin 2012-2013)")

X_tr, X_te = train[FEATS], test[FEATS]
y_tr, y_te = train["target"], test["target"]
d_tr = y_tr - X_tr["wqi"]            # delta =  Next year's WQI − this year's WQI

def imp_pipe(model, scale=False):
    steps = [SimpleImputer(strategy="median")]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)

def new_rf():
    return imp_pipe(RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1))

def new_xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.9,
                        colsample_bytree=0.9, random_state=42)

# (a) Direct models:  Predict the next year's WQI directly.
direct = {"Linear Regression": imp_pipe(LinearRegression(), scale=True),
          "Random Forest": new_rf(), "XGBoost": new_xgb()}
# (b) Delta models:  “Predict the change, then add it to the current WQI.”
delta = {"Linear Regression (delta)": imp_pipe(Ridge(alpha=10.0), scale=True),
         "Random Forest (delta)": new_rf(), "XGBoost (delta)": new_xgb()}

preds = {"Naive (last year's WQI)": X_te["wqi"].values}
for name, m in direct.items():
    m.fit(X_tr, y_tr)
    preds[name] = m.predict(X_te)
for name, m in delta.items():
    m.fit(X_tr, d_tr)
    preds[name] = X_te["wqi"].values + m.predict(X_te)

true_cat = y_te.apply(wqi_class).values
rows = []
for name, p in preds.items():
    err = y_te.values - p
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y_te.values - y_te.mean()) ** 2)
    rows.append({
        "Model": name,
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "R2": 1 - ss_res / ss_tot,
        "Category_Acc_%": np.mean(np.array([wqi_class(v) for v in p]) == true_cat) * 100,
    })
res = pd.DataFrame(rows).sort_values("RMSE").round(3)
print("\n=== Next-year WQI forecast (real Indian river data, time-based test) ===")
print(res.to_string(index=False))
res.to_csv("report/results_real.csv", index=False)

# =====================================================
# 4. Plots + save
# =====================================================
best = res.iloc[0]["Model"]
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_te, preds[best], alpha=0.5, s=18)
lo, hi = min(y_te.min(), preds[best].min()), max(y_te.max(), preds[best].max())
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Actual WQI (next year)"); plt.ylabel("Predicted WQI")
plt.title(f"Best model: {best}")
plt.tight_layout(); plt.savefig("report/figures/real_forecast_scatter.png", dpi=150)
plt.close()

xgb = delta["XGBoost (delta)"]
imp = pd.Series(xgb.feature_importances_, index=FEATS).sort_values()
plt.figure(figsize=(6, 4))
imp.plot.barh()
plt.title("XGBoost (delta) feature importance")
plt.tight_layout(); plt.savefig("report/figures/real_feature_importance.png", dpi=150)
plt.close()

joblib.dump({"direct": direct, "delta": delta, "features": FEATS}, "models/real_models.pkl")
print("\nSaved: report/results_real.csv, data/real_station_year_wqi.csv, models/real_models.pkl")
print("Figures: real_yearly_trend.png, real_forecast_scatter.png, real_feature_importance.png")