from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_report_figures():
    print(f"Loading train-test.csv to generate figures in {FIG_DIR}...")
    train = pd.read_csv(ROOT / "train-test.csv")
    color = "#064A56"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#9DAFB3",
        "axes.labelcolor": "#1F2A2E",
        "xtick.color": "#455A60",
        "ytick.color": "#455A60",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # 1. Target Distribution Histogram
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    ax.hist(train["posted_rate"], bins=60, color=color, alpha=0.88, edgecolor="white", linewidth=0.3)
    ax.set_title("Development-set target distribution")
    ax.set_xlabel("posted_rate ($)")
    ax.set_ylabel("Number of loads")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    target_path = FIG_DIR / "target_distribution.png"
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target_path}")

    # 2. Rate vs Distance Scatter Plot
    sample = train.sample(n=min(8000, len(train)), random_state=42)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    ax.scatter(sample["distance"], sample["posted_rate"], s=8, alpha=0.25, color=color, linewidths=0)
    ax.set_title("posted_rate versus haul distance")
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("posted_rate ($)")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    scatter_path = FIG_DIR / "rate_vs_distance.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {scatter_path}")

    # 3. Missing Values Bar Chart
    missing = train.isna().sum()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    ax.bar(missing.index.astype(str), missing.values, color=color)
    ax.set_title("Missing values in the development set")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    missing_path = FIG_DIR / "missing_values.png"
    fig.savefig(missing_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {missing_path}")

    print("All report figures created successfully!")

if __name__ == "__main__":
    generate_report_figures()
