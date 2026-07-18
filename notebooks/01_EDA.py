"""
EDA for high-rise EPC combined dataset.
Run: jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
Or open in Jupyter / VS Code.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path("..") if Path("data").exists() is False else Path(".")
# Allow running from project root
if not (ROOT / "data" / "processed" / "highrise_combined.csv").exists():
    ROOT = Path(".")

df = pd.read_csv(ROOT / "data" / "processed" / "highrise_combined.csv")
fig_dir = ROOT / "results" / "figures"
table_dir = ROOT / "results" / "tables"
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

print("Shape:", df.shape)
print(df.dtypes)
print(df.head())

# %%
summary = {
    "n_records": len(df),
    "n_residential": int((df["building_typology"] == "residential").sum()),
    "n_commercial": int((df["building_typology"] == "commercial").sum()),
    "target_mean": float(df["energy_consumption"].mean()),
    "target_std": float(df["energy_consumption"].std()),
    "missing_target": int(df["energy_consumption"].isna().sum()),
}
pd.Series(summary).to_csv(table_dir / "eda_summary.csv")
print(summary)

# %%
plt.figure(figsize=(8, 4))
sns.histplot(df["energy_consumption"], bins=50)
plt.title("Target energy consumption distribution")
plt.xlabel("Energy consumption")
plt.tight_layout()
plt.savefig(fig_dir / "eda_target_hist.png", dpi=150)
plt.close()

# %%
plt.figure(figsize=(6, 4))
df["source_city"].value_counts().plot(kind="bar")
plt.title("Records per federated client city")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(fig_dir / "eda_city_counts.png", dpi=150)
plt.close()

# %%
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="source_city", y="energy_consumption")
plt.title("Energy by city (non-IID check)")
plt.tight_layout()
plt.savefig(fig_dir / "eda_energy_by_city.png", dpi=150)
plt.close()

# %%
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="floor_area", y="energy_consumption", hue="building_typology", alpha=0.4)
plt.title("Floor area vs energy")
plt.tight_layout()
plt.savefig(fig_dir / "eda_area_vs_energy.png", dpi=150)
plt.close()

# %%
missing = df.isna().sum().sort_values(ascending=False)
missing[missing > 0].to_csv(table_dir / "eda_missing.csv")
print("Missing values:\n", missing[missing > 0].head(15))

# %%
city_stats = df.groupby("source_city")["energy_consumption"].agg(["count", "mean", "std"])
city_stats.to_csv(table_dir / "eda_city_stats.csv")
print(city_stats)
print("EDA complete — figures in results/figures/")
