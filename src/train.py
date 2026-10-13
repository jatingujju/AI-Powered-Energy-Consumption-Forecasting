
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data import load_and_prepare_data
from src.features import build_training_data, FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.joblib"
COMPARISON_PATH = ROOT / "outputs" / "model_comparison.csv"
METRICS_PATH = ROOT / "outputs" / "metrics.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "test_predictions.csv"


def calculate_metrics(actual, predicted):
    """Calculate regression evaluation metrics."""

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def train_models(hourly):
    """Train and compare models using a chronological split."""

    X, y = build_training_data(hourly)

    # Keep the earliest 80% for training and the latest 20% for testing.
    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTRAINING SUMMARY")
    print("-" * 40)
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")

    # Baseline: predict the same hour as the previous day.
    baseline_predictions = X_test["lag_24"].to_numpy()

    results = [{
        "model": "Previous-day baseline",
        **calculate_metrics(y_test, baseline_predictions),
    }]

    test_predictions = {
        "Previous-day baseline": baseline_predictions,
    }

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    fitted_models = {}

    for name, model in candidates.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Prevent negative energy estimates.
        predictions = predictions.clip(min=0)

        metrics = calculate_metrics(y_test, predictions)

        results.append({
            "model": name,
            **metrics,
        })

        test_predictions[name] = predictions
        fitted_models[name] = model

        print(f"MAE:  {metrics['MAE']:.4f} kWh")
        print(f"RMSE: {metrics['RMSE']:.4f} kWh")
        print(f"R2:   {metrics['R2']:.4f}")

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("RMSE")

    # Select the best fitted ML model using the chronological
    # test RMSE. The baseline remains a separate benchmark.
    ml_results = comparison[
        comparison["model"] != "Previous-day baseline"
    ]

    best_name = ml_results.iloc[0]["model"]
    best_test_rmse = ml_results.iloc[0]["RMSE"]

    # Warn if the best ML model does not beat the baseline.
    baseline_rmse = comparison.loc[
        comparison["model"] == "Previous-day baseline",
        "RMSE",
    ].iloc[0]

    if best_test_rmse >= baseline_rmse:
        print(
            "\nWARNING: The best ML model did not outperform "
            "the previous-day baseline on test RMSE."
        )

    # Save the evaluation predictions before refitting.
    prediction_output = pd.DataFrame({
        "timestamp": y_test.index,
        "actual_kwh": y_test.to_numpy(),
        "baseline_kwh": baseline_predictions,
        "predicted_kwh": test_predictions[best_name],
    })

    ROOT.joinpath("outputs").mkdir(
        parents=True, exist_ok=True
    )
    ROOT.joinpath("models").mkdir(
        parents=True, exist_ok=True
    )

    comparison.to_csv(COMPARISON_PATH, index=False)
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    selected_metrics = comparison[
        comparison["model"] == best_name
    ]

    selected_metrics.to_csv(METRICS_PATH, index=False)

    # Refit the selected estimator on all usable historical rows
    # for future forecasting. Test metrics remain from the
    # held-out chronological test period.
    final_model = candidates[best_name]
    final_model.fit(X, y)

    joblib.dump({
        "model": final_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
    }, MODEL_PATH)

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print(f"\nSelected ML model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")

    return comparison, prediction_output


if __name__ == "__main__":
    hourly_data = load_and_prepare_data()
    train_models(hourly_data)
