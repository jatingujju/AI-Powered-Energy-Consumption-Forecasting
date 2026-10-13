
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_and_prepare_data

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "images"


def save_consumption_trend(hourly):
    recent = hourly.tail(24 * 14)

    plt.figure(figsize=(13, 5))
    plt.plot(
        recent.index,
        recent["energy_kwh"],
        linewidth=1.2,
    )
    plt.title("Household Energy Consumption — Last 14 Days")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy (kWh per hour)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "consumption_trend.png", dpi=160)
    plt.close()


def save_actual_vs_predicted():
    path = OUTPUT_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Test predictions not found. Run python -m src.train first."
        )

    predictions = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "actual_kwh", "predicted_kwh"}
    if not required.issubset(predictions.columns):
        raise ValueError(
            f"Expected prediction columns: {sorted(required)}. "
            f"Found: {list(predictions.columns)}"
        )

    # Plot a representative sample of the held-out test period.
    sample = predictions.tail(24 * 7)

    plt.figure(figsize=(13, 5))
    plt.plot(
        sample["timestamp"],
        sample["actual_kwh"],
        label="Actual",
        linewidth=1.5,
    )
    plt.plot(
        sample["timestamp"],
        sample["predicted_kwh"],
        label="Predicted",
        linewidth=1.5,
    )
    plt.title("Actual vs. Predicted Energy — Test Period")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy (kWh per hour)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()


def save_model_comparison():
    path = OUTPUT_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Model comparison not found. Run python -m src.train first."
        )

    comparison = pd.read_csv(path).sort_values("RMSE")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(comparison["model"], comparison["RMSE"])
    plt.bar_label(bars, fmt="%.3f", padding=3)
    plt.title("Model Comparison — RMSE (Lower Is Better)")
    plt.xlabel("Model")
    plt.ylabel("RMSE (kWh)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=160)
    plt.close()


def save_forecast_chart():
    path = OUTPUT_DIR / "next_24_hours.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Forecast not found. Run python -m src.forecast first."
        )

    forecast = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "predicted_kwh"}
    if not required.issubset(forecast.columns):
        raise ValueError(
            f"Expected forecast columns: {sorted(required)}. "
            f"Found: {list(forecast.columns)}"
        )

    plt.figure(figsize=(13, 5))
    plt.plot(
        forecast["timestamp"],
        forecast["predicted_kwh"],
        marker="o",
        linewidth=1.8,
    )
    plt.title("Next 24 Hours — Predicted Energy Consumption")
    plt.xlabel("Forecast timestamp")
    plt.ylabel("Predicted energy (kWh per hour)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "next_24_hours_forecast.png", dpi=160)
    plt.close()


def save_cost_simulation(forecast):
    # Illustrative assumption only; replace with the applicable tariff.
    tariff_per_kwh = 8.0

    simulation = forecast[["timestamp", "predicted_kwh"]].copy()
    simulation["assumed_tariff_inr_per_kwh"] = tariff_per_kwh
    simulation["estimated_cost_inr"] = (
        simulation["predicted_kwh"] * tariff_per_kwh
    )

    output_path = OUTPUT_DIR / "business_simulation.csv"
    simulation.to_csv(output_path, index=False)

    print("\nILLUSTRATIVE COST SIMULATION")
    print("-" * 45)
    print("Assumed tariff: Rs. 8.00 per kWh")
    print(
        "Estimated 24-hour cost: "
        f"Rs. {simulation['estimated_cost_inr'].sum():.2f}"
    )
    print("This is a simulation, not an actual electricity bill.")
    print(f"Saved to: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading historical energy data...")
    hourly = load_and_prepare_data()

    print("Creating consumption trend chart...")
    save_consumption_trend(hourly)

    print("Creating actual-versus-predicted chart...")
    save_actual_vs_predicted()

    print("Creating model comparison chart...")
    save_model_comparison()

    print("Creating 24-hour forecast chart...")
    save_forecast_chart()

    forecast = pd.read_csv(
        OUTPUT_DIR / "next_24_hours.csv",
        parse_dates=["timestamp"],
    )
    save_cost_simulation(forecast)

    print("\nVISUALIZATION COMPLETE")
    print("-" * 45)
    print("Generated images:")
    for filename in [
        "consumption_trend.png",
        "actual_vs_predicted.png",
        "model_comparison.png",
        "next_24_hours_forecast.png",
    ]:
        print(IMAGE_DIR / filename)


if __name__ == "__main__":
    main()

