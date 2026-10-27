import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wqi import wqi_category

st.set_page_config(page_title="Water Quality Forecast", page_icon="💧", layout="wide")

st.title("💧 AI Water Quality Forecasting System")
tab_real, tab_syn = st.tabs(["🇮🇳 Real Data: Indian Rivers (next-year forecast)",
                             "🧪 Synthetic Data: 7-day forecast prototype"])


# =====================================================================
# TAB 1: REAL DATA (CPCB-style yearly station data)
# =====================================================================
REAL_FEATS_NOTE = "Real WQI scale: 0-100, HIGHER = BETTER water quality."


def real_class(v):
    if v >= 90:
        return "Excellent"
    if v >= 70:
        return "Good"
    if v >= 50:
        return "Medium"
    if v >= 25:
        return "Bad"
    return "Very Bad"


@st.cache_data
def load_real():
    df = pd.read_csv("data/real_station_year_wqi.csv", dtype={"station": str})
    df["state"] = df["state"].fillna("Unknown")
    df["location"] = df["location"].fillna("(location not available)")
    df["log_tc"] = np.log1p(df["tc"])
    df["log_fecal"] = np.log1p(df["fecal_coliform"])
    prev = df[["station", "year", "wqi"]].copy()
    prev["year"] += 1
    prev = prev.rename(columns={"wqi": "wqi_prev"})
    df = df.merge(prev, on=["station", "year"], how="left")
    df["wqi_change"] = df["wqi"] - df["wqi_prev"]
    return df


@st.cache_resource
def load_real_models():
    return joblib.load("models/real_models.pkl")


def real_status(category):
    if category in ("Excellent", "Good"):
        st.success(f"✅ Status: {category}. Water quality is within acceptable limits.")
    elif category == "Medium":
        st.warning(f"⚠️ Status: {category}. Treatment and regular monitoring recommended.")
    else:
        st.error(f"🚨 Status: {category}. ALERT: water is unfit for use without major treatment.")


def real_tab():
    try:
        rdf = load_real()
        bundle = load_real_models()
    except FileNotFoundError:
        st.info("Real-data files not found. Run `python src\\real_data_forecast.py` first.")
        return

    feats = bundle["features"]
    all_models = {**bundle["direct"], **bundle["delta"]}
    is_delta = {name: True for name in bundle["delta"]}

    st.caption("Next-year Water Quality Index (WQI) forecast for Indian river monitoring stations "
               "(yearly station averages, 2003-2014). " + REAL_FEATS_NOTE)

    # ---------- Controls ----------
    c1, c2, c3, c4 = st.columns([1.2, 2, 0.8, 1.5])
    state = c1.selectbox("State", sorted(rdf["state"].unique()))
    sdf = rdf[rdf["state"] == state]
    labels = (sdf.drop_duplicates("station")
                 .assign(label=lambda x: x["station"] + " | " + x["location"].str.slice(0, 60)))
    label_to_station = dict(zip(labels["label"], labels["station"]))
    station_label = c2.selectbox("Station", sorted(label_to_station))
    station = label_to_station[station_label]
    hist = rdf[rdf["station"] == station].sort_values("year")
    years = sorted(hist["year"].unique())
    year = c3.selectbox("Base year", years, index=len(years) - 1)
    model_name = c4.selectbox("Model", list(all_models), index=list(all_models).index("Linear Regression (delta)"))

    # ---------- Prediction ----------
    row = hist[hist["year"] == year].iloc[0]
    X = pd.DataFrame([row[feats].astype(float)])
    model = all_models[model_name]
    raw_pred = float(model.predict(X)[0])
    pred = row["wqi"] + raw_pred if model_name in is_delta else raw_pred
    pred = float(np.clip(pred, 0, 100))
    current = float(row["wqi"])
    next_year = int(year) + 1
    actual_row = hist[hist["year"] == next_year]
    actual = float(actual_row["wqi"].iloc[0]) if len(actual_row) else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"WQI in {year}", f"{current:.1f}", real_class(current), delta_color="off")
    m2.metric(f"Predicted WQI {next_year}", f"{pred:.1f}", f"{pred - current:+.1f}")
    m3.metric("Predicted Category", real_class(pred))
    if not np.isnan(actual):
        m4.metric(f"Actual WQI {next_year}", f"{actual:.1f}", f"error {pred - actual:+.1f}", delta_color="off")
    else:
        m4.metric(f"Actual WQI {next_year}", "Not in dataset")

    real_status(real_class(pred))
    st.caption(f"Station: **{row['location']}** ({row['state']}). "
               f"Years of data for this station: {len(years)}.")

    # ---------- Chart ----------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["year"], y=hist["wqi"], mode="lines+markers",
                             name="WQI (history)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=[year, next_year], y=[current, pred], mode="lines+markers",
                             name="Forecast", line=dict(color="red", dash="dash")))
    if not np.isnan(actual):
        fig.add_trace(go.Scatter(x=[next_year], y=[actual], mode="markers", name="Actual",
                                 marker=dict(color="green", size=12, symbol="star")))
    for y, label in [(90, "Excellent"), (70, "Good"), (50, "Medium"), (25, "Bad")]:
        fig.add_hline(y=y, line_dash="dot", line_color="gray", annotation_text=label)
    fig.update_layout(height=380, yaxis_title="WQI (higher = better)", xaxis_title="Year",
                      yaxis=dict(range=[0, 105]), xaxis=dict(dtick=1), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    # ---------- Readings + model comparison ----------
    left, right = st.columns(2)
    with left:
        st.subheader(f"Readings in {year}")
        names = {"temperature": "Temperature (°C)", "do": "Dissolved oxygen (mg/l)", "ph": "pH",
                 "ec": "Conductivity (µmhos/cm)", "bod": "BOD (mg/l)", "nitrate": "Nitrate (mg/l)",
                 "fecal_coliform": "Fecal coliform (MPN/100ml)", "tc": "Total coliform (MPN/100ml)",
                 "wqi": "WQI"}
        table = pd.DataFrame({"value": [row[k] for k in names]}, index=list(names.values()))
        st.dataframe(table.round(2))
    with right:
        st.subheader("Model comparison (test: 2012-2013 → 2013-2014)")
        try:
            st.dataframe(pd.read_csv("report/results_real.csv"), hide_index=True)
        except FileNotFoundError:
            st.info("report/results_real.csv not found.")

    # ---------- Overview ----------
    st.subheader("All monitored stations: mean WQI by year")
    trend = rdf.groupby("year")["wqi"].agg(["mean", "count"]).reset_index()
    f2 = go.Figure(go.Scatter(x=trend["year"], y=trend["mean"], mode="lines+markers",
                              text=trend["count"], hovertemplate="Year %{x}<br>Mean WQI %{y:.1f}<br>Stations %{text}"))
    f2.update_layout(height=300, yaxis_title="Mean WQI", xaxis=dict(dtick=1), margin=dict(t=10))
    st.plotly_chart(f2, width="stretch")

    st.caption("Data: Kaggle 'Indian water quality data' (government-sourced yearly station averages). "
               "Yearly averages and many missing values limit accuracy; a simple last-year baseline is hard to beat.")


# =====================================================================
# TAB 2: SYNTHETIC DATA (7-day forecast prototype)
# =====================================================================
COLS = ["temperature", "ph", "dissolved_oxygen", "turbidity", "conductivity", "bod", "wqi"]
H = 7


@st.cache_data
def load_data():
    df = pd.read_csv("data/water_quality_wqi.csv", parse_dates=["date"], index_col="date")
    return df.asfreq("D")


@st.cache_resource
def load_model():
    return joblib.load("models/xgb_wqi.pkl")


def build_features(df):
    feat = pd.DataFrame(index=df.index)
    for c in COLS:
        for lag in [0, 1, 2, 3, 7, 14]:
            feat[f"{c}_lag{lag}"] = df[c].shift(lag)
        feat[f"{c}_roll7"] = df[c].rolling(7).mean()
    doy = df.index.day_of_year
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return feat


def show_status(category):
    if category in ("Excellent", "Good"):
        st.success(f"✅ Status: {category}. Water quality is within acceptable limits.")
    elif category == "Poor":
        st.warning(f"⚠️ Status: {category}. Using untreated water is risky; increase monitoring.")
    else:
        st.error(f"🚨 Status: {category}. ALERT: water is unfit for use.")


def synthetic_tab():
    st.caption("7-day ahead Water Quality Index (WQI) forecast with decision support alerts (XGBoost). "
               "Synthetic daily data (Station_A). Synthetic WQI scale: HIGHER = WORSE water quality.")
    df = load_data()
    bundle = load_model()
    model, feat_cols = bundle["model"], bundle["features"]
    feat = build_features(df)

    min_d = df.index[14].date()
    max_d = df.index[-1].date()
    c1, c2 = st.columns([1, 2])
    origin = pd.Timestamp(c1.date_input("Forecast date (predicts 7 days ahead)",
                                        value=max_d, min_value=min_d, max_value=max_d))
    window = c2.slider("History window (days)", 30, 365, 120)

    x = feat.loc[[origin], feat_cols]
    pred = float(model.predict(x)[0])
    target_date = origin + pd.Timedelta(days=H)
    current = float(df.loc[origin, "wqi"])
    actual = df["wqi"].get(target_date, np.nan)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Current WQI ({origin.date()})", f"{current:.1f}", wqi_category(current))
    m2.metric(f"Predicted WQI ({target_date.date()})", f"{pred:.1f}", f"{pred - current:+.1f}", delta_color="inverse")
    m3.metric("Predicted Category", wqi_category(pred))
    if not np.isnan(actual):
        m4.metric("Actual WQI (from data)", f"{actual:.1f}", f"error {pred - actual:+.1f}", delta_color="off")
    else:
        m4.metric("Actual WQI", "Not available (future date)")

    show_status(wqi_category(pred))

    hist = df["wqi"].loc[origin - pd.Timedelta(days=window): origin]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="WQI (history)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=[origin, target_date], y=[current, pred], name="Forecast",
                             mode="lines+markers", line=dict(color="red", dash="dash")))
    if not np.isnan(actual):
        fig.add_trace(go.Scatter(x=[target_date], y=[actual], name="Actual", mode="markers",
                                 marker=dict(color="green", size=11, symbol="star")))
    for y, label in [(50, "Excellent/Good"), (100, "Good/Poor"), (200, "Poor/Very Poor")]:
        fig.add_hline(y=y, line_dash="dot", line_color="gray", annotation_text=label)
    fig.update_layout(height=420, yaxis_title="WQI (higher = worse)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Readings on {origin.date()}")
        st.dataframe(df.loc[[origin], COLS].T.rename(columns={origin: "value"}).round(2))
    with right:
        st.subheader("Model comparison (test set)")
        try:
            st.dataframe(pd.read_csv("report/results.csv"), hide_index=True)
        except FileNotFoundError:
            st.info("report/results.csv not found. Run forecast.py first.")

    st.caption("Note: Demo data is synthetic (Station_A). The model can be retrained on real river data.")


with tab_real:
    real_tab()
with tab_syn:
    synthetic_tab()