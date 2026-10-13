
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_and_prepare_data
from src.features import make_features, FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.joblib"
FORECAST_PATH = ROOT / "outputs" / "next_24_hours.csv"


def forecast_next_24_hours(hourly):
    """Generate recursive hourly energy predictions for 24 hours."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python -m src.train first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

    history = hourly["energy_kwh"].astype(float).copy()
    history = history.sort_index()

    if len(history) < 168:
        raise ValueError(
            "At least 168 hours of historical data is required."
        )

    future_times = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
    )

    predictions = []

    for timestamp in future_times:
        # Add the future timestamp without an observed target.
        extended = pd.concat([
            history,
            pd.Series([float("nan")], index=[timestamp]),
        ])

        # Build features using historical values only.
        X_future = make_features(extended).iloc[[-1]]

        X_future = X_future[FEATURE_COLUMNS]

        if X_future.isna().any().any():
            raise ValueError(
                f"Missing features for forecast timestamp {timestamp}."
            )

        # Predict one hour ahead.
        predicted_kwh = float(
            model.predict(X_future)[0]
        )

        # Electricity consumption cannot be negative.
        predicted_kwh = max(0.0, predicted_kwh)

        predictions.append(predicted_kwh)

        # Use the prediction as history for the next forecast.
        history.loc[timestamp] = predicted_kwh

    forecast = pd.DataFrame({
        "timestamp": future_times,
        "predicted_kwh": predictions,
    })

    FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(FORECAST_PATH, index=False)

    print("\nNEXT 24 HOURS FORECAST")
    print("-" * 60)
    print(forecast.to_string(index=False))

    print("\nFORECAST SUMMARY")
    print("-" * 40)
    print(f"Forecast hours: {len(forecast)}")
    print(
        f"Total predicted energy: "
        f"{forecast['predicted_kwh'].sum():.2f} kWh"
    )
    print(
        f"Average hourly energy: "
        f"{forecast['predicted_kwh'].mean():.2f} kWh"
    )
    print(
        f"Peak predicted hourly energy: "
        f"{forecast['predicted_kwh'].max():.2f} kWh"
    )
    print(f"Saved to: {FORECAST_PATH}")

    return forecast


if __name__ == "__main__":
    hourly_data = load_and_prepare_data()
    forecast_next_24_hours(hourly_data)
