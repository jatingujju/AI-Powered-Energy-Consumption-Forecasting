
import argparse
import sys

from src.data import load_and_prepare_data
from src.features import build_training_data
from src.train import train_models
from src.forecast import forecast_next_24_hours
from src.visualize import main as run_visualizations


def prepare():
    """Prepare and save the hourly energy dataset."""
    hourly = load_and_prepare_data()
    print(f"\nPreparation complete: {len(hourly):,} hourly records.")
    return hourly


def train():
    """Train and evaluate forecasting models."""
    hourly = load_and_prepare_data()
    train_models(hourly)


def forecast():
    """Generate the next 24-hour forecast."""
    hourly = load_and_prepare_data()
    forecast_next_24_hours(hourly)


def visualize():
    """Generate charts and the illustrative cost simulation."""
    run_visualizations()


def run_all():
    """Execute the complete pipeline in order."""
    print("\n========== 1. DATA PREPARATION ==========")
    prepare()

    print("\n========== 2. MODEL TRAINING ==========")
    train()

    print("\n========== 3. 24-HOUR FORECAST ==========")
    forecast()

    print("\n========== 4. VISUALIZATIONS ==========")
    visualize()

    print("\nCOMPLETE PIPELINE FINISHED SUCCESSFULLY.")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Energy Consumption Forecasting System"
    )

    parser.add_argument(
        "command",
        choices=["prepare", "train", "forecast", "visualize", "all"],
        help="Pipeline stage to execute",
    )

    args = parser.parse_args()

    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "train":
            train()
        elif args.command == "forecast":
            forecast()
        elif args.command == "visualize":
            visualize()
        elif args.command == "all":
            run_all()

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        print(
            "Check the error details above before rerunning.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

