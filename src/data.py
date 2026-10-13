
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "household_power_consumption.txt"
)

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_energy.csv"
)


def load_and_prepare_data():
    """Load, clean and aggregate electricity data to hourly kWh."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    print("Loading the electricity dataset...")

    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    print(f"Original rows loaded: {len(df):,}")

    # Combine date and time into a timestamp.
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce",
    )

    # Remove invalid timestamps and duplicate observations.
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp")

    # Keep valid non-negative active power readings.
    power = df["Global_active_power"].where(
        df["Global_active_power"] >= 0
    )

    print("Aggregating minute readings into hourly energy...")

    # Require at least 45 valid readings per hour.
    # Mean power (kW) multiplied by one hour gives kWh.
    hourly_energy = power.resample("h").agg(
        lambda values: (
            values.mean()
            if values.count() >= 45
            else float("nan")
        )
    )

    hourly = hourly_energy.to_frame(
        name="energy_kwh"
    )

    hourly.index.name = "timestamp"

    # Interpolate only short internal gaps of up to two hours.
    hourly["energy_kwh"] = hourly["energy_kwh"].interpolate(
        method="time",
        limit=2,
        limit_area="inside",
    )

    hourly = hourly.dropna()
    hourly = hourly[hourly["energy_kwh"] >= 0]

    if len(hourly) < 24 * 14:
        raise ValueError(
            "Insufficient valid hourly data."
        )

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_csv(PROCESSED_FILE)

    print("\nDATA PREPARATION SUMMARY")
    print("-" * 40)
    print(f"Valid hourly rows: {len(hourly):,}")
    print(f"Start timestamp: {hourly.index.min()}")
    print(f"End timestamp: {hourly.index.max()}")
    print(f"Missing values: {hourly.isna().sum().sum()}")
    print(f"Saved to: {PROCESSED_FILE}")

    print("\nFirst five hourly records:")
    print(hourly.head())

    return hourly


if __name__ == "__main__":
    load_and_prepare_data()
