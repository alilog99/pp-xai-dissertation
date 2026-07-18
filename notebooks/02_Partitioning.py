"""
Federated geographic partitioning visualisation (non-IID check).
Run from project root:
  source scripts/env.sh && python notebooks/02_Partitioning.py
Or open notebooks/02_Partitioning.ipynb in Jupyter / VS Code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path("..") if not Path("data/federated").exists() else Path(".")
if not (ROOT / "data" / "federated" / "client_london.csv").exists():
    ROOT = Path(".")

fig_dir = ROOT / "results" / "figures"
table_dir = ROOT / "results" / "tables"
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

clients = ["london", "manchester", "birmingham"]
frames = []
for city in clients:
    path = ROOT / "data" / "federated" / f"client_{city}.csv"
    part = pd.read_csv(path)
    part["client"] = city
    frames.append(part)
    print(f"{city}: n={len(part)}, mean_energy={part['energy_consumption'].mean():.2f}")

df = pd.concat(frames, ignore_index=True)

stats = pd.read_csv(table_dir / "partition_stats.csv")
print("\npartition_stats.csv:\n", stats)

# Recompute / extend stats for dissertation tables
detailed = (
    df.groupby("client")
    .agg(
        records=("energy_consumption", "count"),
        mean_energy=("energy_consumption", "mean"),
        std_energy=("energy_consumption", "std"),
        median_energy=("energy_consumption", "median"),
        mean_floor_area=("floor_area", "mean"),
        pct_commercial=("building_typology", lambda s: 100.0 * (s == "commercial").mean()),
    )
    .reset_index()
)
detailed.to_csv(table_dir / "partition_detailed_stats.csv", index=False)
print("\nDetailed stats:\n", detailed)

# Bar: client sizes
plt.figure(figsize=(6, 4))
sns.barplot(data=stats, x="client", y="records", color="#4C72B0")
plt.title("Federated client sizes (geographic partition)")
plt.ylabel("Number of records")
plt.xlabel("Client (city)")
plt.tight_layout()
plt.savefig(fig_dir / "partition_client_sizes.png", dpi=150)
plt.close()

# Bar: mean energy (label skew)
plt.figure(figsize=(6, 4))
sns.barplot(data=stats, x="client", y="mean_energy", color="#55A868")
plt.title("Mean energy consumption by client (mild non-IID)")
plt.ylabel("Mean energy consumption")
plt.xlabel("Client (city)")
plt.tight_layout()
plt.savefig(fig_dir / "partition_mean_energy.png", dpi=150)
plt.close()

# Boxplot: energy distributions — dissertation fig8
plt.figure(figsize=(8, 4.5))
order = ["london", "manchester", "birmingham"]
sns.boxplot(data=df, x="client", y="energy_consumption", order=order)
plt.title("Client energy distributions (non-IID geographic split)")
plt.ylabel("Energy consumption")
plt.xlabel("Federated client")
plt.tight_layout()
plt.savefig(fig_dir / "fig8_client_distributions.png", dpi=150)
plt.savefig(fig_dir / "partition_energy_boxplot.png", dpi=150)
plt.close()

# Typology mix per client
plt.figure(figsize=(7, 4))
mix = df.groupby(["client", "building_typology"]).size().reset_index(name="count")
sns.barplot(data=mix, x="client", y="count", hue="building_typology", order=order)
plt.title("Building typology mix by federated client")
plt.ylabel("Count")
plt.xlabel("Client (city)")
plt.tight_layout()
plt.savefig(fig_dir / "partition_typology_mix.png", dpi=150)
plt.close()

# Non-IID summary for console / viva
overall_mean = df["energy_consumption"].mean()
max_rel = ((stats["mean_energy"] - overall_mean).abs() / overall_mean * 100).max()
print(f"\nOverall mean energy: {overall_mean:.2f}")
print(f"Max |client mean − overall| / overall: {max_rel:.2f}%")
print("Partitioning visualisation complete — figures in results/figures/")
