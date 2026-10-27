import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/water_quality_clean.csv", parse_dates=["date"], index_col="date")

# Weighted Arithmetic WQI
# standard (Si) = permissible limit, ideal (Vo) = ideal value
# Qi = |Vi - Vo| / (Si - Vo) * 100,  Wi = 1/Si,  WQI = sum(Wi*Qi) / sum(Wi)
params = {
    "ph":               {"standard": 8.5,  "ideal": 7.0},
    "dissolved_oxygen": {"standard": 5.0,  "ideal": 14.6},
    "turbidity":        {"standard": 100,  "ideal": 0},
    "conductivity":     {"standard": 750,  "ideal": 0},
    "bod":              {"standard": 3.0,  "ideal": 0},
}


def compute_wqi(data):
    num, den = 0, 0
    for col, p in params.items():
        qi = (data[col] - p["ideal"]).abs() / abs(p["standard"] - p["ideal"]) * 100
        wi = 1 / p["standard"]
        num = num + wi * qi
        den += wi
    return num / den


def wqi_category(w):
    if w <= 50:
        return "Excellent"
    elif w <= 100:
        return "Good"
    elif w <= 200:
        return "Poor"
    elif w <= 300:
        return "Very Poor"
    return "Unsafe"


if __name__ == "__main__":
    df["wqi"] = compute_wqi(df)
    df["category"] = df["wqi"].apply(wqi_category)
    df.to_csv("data/water_quality_wqi.csv")

    print(df[["wqi", "category"]].describe(include="all"))
    print("\nCategory counts:\n", df["category"].value_counts())

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["wqi"], linewidth=0.8)
    plt.axhline(50, color="green", linestyle="--", label="Excellent / Good")
    plt.axhline(100, color="orange", linestyle="--", label="Good / Poor")
    plt.axhline(200, color="red", linestyle="--", label="Poor / Very Poor")
    plt.title("Water Quality Index Over Time")
    plt.ylabel("WQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/figures/wqi.png", dpi=150)
    plt.show()