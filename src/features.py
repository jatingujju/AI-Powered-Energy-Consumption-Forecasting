
import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]


def make_features(series: pd.Series) -> pd.DataFrame:
    """Create forecasting features using historical observations."""

    series = series.sort_index().astype(float)
    X = pd.DataFrame(index=series.index)

    # Calendar features
    X["hour"] = series.index.hour
    X["day_of_week"] = series.index.dayofweek
    X["day_of_year"] = series.index.dayofyear
    X["is_weekend"] = (
        series.index.dayofweek >= 5
    ).astype(int)

    # Historical consumption features
    X["lag_1"] = series.shift(1)
    X["lag_24"] = series.shift(24)
    X["lag_168"] = series.shift(168)

    # Mean of the previous 24 hours, excluding current hour
    X["rolling_mean_24"] = (
        series.shift(1).rolling(window=24).mean()
    )

    return X[FEATURE_COLUMNS]


def build_training_data(hourly: pd.DataFrame):
    """Create aligned features and target values."""

    if "energy_kwh" not in hourly.columns:
        raise ValueError("Missing required column: energy_kwh")

    if not isinstance(hourly.index, pd.DatetimeIndex):
        raise TypeError("A DatetimeIndex is required.")

    hourly = hourly.sort_index()
    series = hourly["energy_kwh"].astype(float)

    X = make_features(series)
    y = series.rename("actual_kwh")

    combined = X.join(y).dropna()

    if combined.empty:
        raise ValueError("No usable training records found.")

    X = combined[FEATURE_COLUMNS]
    y = combined["actual_kwh"]

    print("\nFEATURE ENGINEERING SUMMARY")
    print("-" * 40)
    print(f"Usable records: {len(X):,}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Missing feature values: {X.isna().sum().sum()}")
    print(f"Missing target values: {y.isna().sum()}")

    return X, y


if __name__ == "__main__":
    from src.data import load_and_prepare_data

    hourly_data = load_and_prepare_data()
    X, y = build_training_data(hourly_data)

    print("\nFIRST FIVE FEATURE ROWS")
    print(X.head())

    print("\nFIRST FIVE TARGET VALUES")
    print(y.head())
