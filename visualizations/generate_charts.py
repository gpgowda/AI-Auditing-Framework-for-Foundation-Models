"""
Final AI Audit Model Comparison Visualization

Reads:
    reports/comparison_table.csv

Creates ONE main comparison chart showing:
    - Truthfulness
    - Safety
    - Robustness
    - Bias
    - Privacy
    - Transparency

All selected metrics are oriented so that:
    Higher score = better performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "reports/comparison_table.csv"
OUTPUT_FILE = "visualizations/comparison_chart.png"


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. "
        "Run the final aggregation first."
    )

df = pd.read_csv(INPUT_FILE)


# ============================================================
# SELECT ONE REPRESENTATIVE METRIC PER DIMENSION
# ============================================================

dimension_metrics = {
    "Truthfulness": "TA",
    "Safety": "HRR",
    "Robustness": "RS",
    "Bias": "DOPS",
    "Privacy": "privacy_compliance_rate",
    "Transparency": "EQS",
}


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["model"] + list(dimension_metrics.values())

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# MODEL ORDER
# ============================================================

model_order = ["GEMINI", "GPT", "LLAMA"]

df["model"] = pd.Categorical(
    df["model"],
    categories=model_order,
    ordered=True
)

df = df.sort_values("model")


# ============================================================
# PREPARE DATA
# ============================================================

dimensions = list(dimension_metrics.keys())

plot_data = pd.DataFrame()

for dimension, metric in dimension_metrics.items():
    plot_data[dimension] = df[metric].astype(float).values

plot_data.index = df["model"].astype(str)


# ============================================================
# CREATE ONE COMPARISON CHART
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

plot_data.plot(
    kind="bar",
    ax=ax,
    width=0.75
)


# ============================================================
# FORMAT CHART
# ============================================================

ax.set_title(
    "AI Audit Performance Comparison Across Six Dimensions",
    fontsize=14
)

ax.set_xlabel("Model", fontsize=11)

ax.set_ylabel(
    "Audit Score (higher = better)",
    fontsize=11
)

ax.set_ylim(0, 1.05)

ax.set_xticklabels(
    ["Gemini", "GPT", "Llama"],
    rotation=0
)

ax.legend(
    title="Audit Dimension",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


# ============================================================
# ADD VALUE LABELS
# ============================================================

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=2,
        fontsize=8
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("FINAL COMPARISON CHART GENERATED")
print("=" * 60)

print(f"Saved to:")
print(f"  {OUTPUT_FILE}")

print()
print("Dimensions included:")

for dimension, metric in dimension_metrics.items():
    print(f"  {dimension}: {metric}")

print()
print("Higher score = better performance")
print("=" * 60)