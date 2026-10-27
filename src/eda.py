import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("report/figures", exist_ok=True)
df = pd.read_csv("data/water_quality_clean.csv", parse_dates=["date"], index_col="date")

# 1. Trend plots
df.plot(subplots=True, figsize=(12, 12), title="Water Quality Parameters Over Time")
plt.tight_layout()
plt.savefig("report/figures/trends.png", dpi=150)

# 2. Seasonality (monthly boxplot)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), df.columns):
    sns.boxplot(x=df.index.month, y=df[col], ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Month")
plt.tight_layout()
plt.savefig("report/figures/seasonality.png", dpi=150)

# 3. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation")
plt.tight_layout()
plt.savefig("report/figures/correlation.png", dpi=150)

plt.show()