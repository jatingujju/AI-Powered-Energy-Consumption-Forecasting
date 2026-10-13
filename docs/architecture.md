# System Architecture
## AI-Powered Energy Consumption Forecasting System

## 1. Overview

This project implements an end-to-end machine learning pipeline for forecasting household electricity consumption using historical data from the UCI Machine Learning Repository.

The system prepares historical data, engineers forecasting features, trains and evaluates machine learning models, generates a 24-hour forecast, and produces visualizations and an illustrative electricity-cost report.

The system uses an offline public dataset. It does not connect to live smart meters or industrial equipment.

## 2. Architecture Diagram

```text
          Public Household Electricity Dataset
                         |
                         v
                   data/raw/
                         |
                         v
                  +--------------+
                  |  src/data.py |
                  | Data Cleaning|
                  | Hourly Data  |
                  +--------------+
                         |
                         v
             data/processed/hourly_energy.csv
                         |
                         v
                +------------------+
                | src/features.py  |
                | Feature Creation |
                +------------------+
                         |
                         v
                   Training Data
                         |
                         v
                 +----------------+
                 |  src/train.py  |
                 | Train and Test |
                 | Model Comparison|
                 +----------------+
                         |
                         v
                 models/best_model.joblib
                         |
                         v
                +------------------+
                | src/forecast.py  |
                | 24-Hour Forecast |
                +------------------+
                         |
                         v
              outputs/next_24_hours.csv
                         |
                         v
                +------------------+
                | src/visualize.py |
                | Charts and Costs |
                +------------------+
                         |
              +----------+----------+
              |                     |
              v                     v
          images/                outputs/
        Four Charts          CSV Reports
```

## 3. Main Components

### 3.1 Data Preparation — `src/data.py`

Responsibilities:

- Load the original household electricity dataset.
- Parse date and time information.
- Convert electricity readings to numeric values.
- Handle missing and invalid readings.
- Aggregate valid minute-level active power readings into hourly energy estimates.
- Interpolate eligible short gaps.
- Save the processed dataset to `data/processed/hourly_energy.csv`.

The processed dataset currently contains 34,173 valid hourly records.

### 3.2 Feature Engineering — `src/features.py`

The feature-engineering module prepares the input variables used by the forecasting models.

It creates eight features:

- Hour of day
- Day of week
- Day of year
- Weekend indicator
- Previous-hour consumption
- Previous-day consumption
- Previous-week consumption
- Rolling mean of the previous 24 hours

The resulting feature dataset is aligned with the target consumption values and rows with unavailable feature values are excluded.

### 3.3 Model Training — `src/train.py`

The training module evaluates three approaches:

- Previous-day baseline
- Linear Regression
- Random Forest Regressor

The observations are divided chronologically into training and testing periods.

The module calculates MAE, RMSE and R², compares model performance, selects the best-performing machine-learning model according to the implemented selection rule, and saves the trained model.

Generated files include:

- `models/best_model.joblib`
- `outputs/metrics.csv`
- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`

The reported model results are preliminary. A separate validation period and an untouched final test set would provide stronger evidence of generalization performance.

### 3.4 Forecasting — `src/forecast.py`

The forecasting module loads the trained model and historical hourly energy data.

It generates predictions for the next 24 hourly periods. Each predicted value is added to the history before the next forecast is generated, so subsequent predictions use earlier forecast values where required by the features.

The resulting forecast is saved to:

`outputs/next_24_hours.csv`

Because the source dataset ends in November 2010, the current example forecast also starts from that historical endpoint. It does not represent current electricity consumption.

### 3.5 Visualization and Cost Simulation — `src/visualize.py`

The visualization module creates four charts:

1. Historical consumption trend over the last 14 days of available data.
2. Actual versus predicted consumption on the held-out test period.
3. Model RMSE comparison.
4. Next 24-hour energy forecast.

The charts are saved in the `images/` directory.

The module also creates `outputs/business_simulation.csv`, which estimates a 24-hour electricity cost using an assumed flat tariff of ₹8 per kWh.

This cost is illustrative and is not an actual electricity bill.

### 3.6 Application Entry Point — `main.py`

The main runner provides a command-line interface for executing individual stages or the entire pipeline.

Supported commands:

- `python main.py prepare`
- `python main.py train`
- `python main.py forecast`
- `python main.py visualize`
- `python main.py all`

The `all` command runs preparation, training, forecasting, and visualization in sequence.

## 4. Data and Artifact Storage

| Directory | Purpose |
|---|---|
| `data/raw/` | Original dataset |
| `data/processed/` | Prepared hourly data |
| `src/` | Pipeline source code |
| `models/` | Saved trained model |
| `outputs/` | Forecasts, predictions, metrics and cost report |
| `images/` | Generated visualizations |
| `docs/` | Project documentation |

## 5. Technology Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 6. Current Limitations

- The dataset contains historical household data rather than industrial or live sensor measurements.
- The original observations end in November 2010.
- Missing-hour handling should be strengthened so lag features always correspond to the intended elapsed time.
- Model selection and evaluation should be improved with a separate validation period and a final untouched test set.
- Recursive forecasts can accumulate errors over the forecast horizon.
- The cost estimate uses an assumed flat tariff and omits other electricity-bill charges.

## 7. Future Improvements

- Add explicit hourly reindexing and robust gap handling.
- Introduce walk-forward time-series validation.
- Compare against seasonal-naive forecasting baselines.
- Develop an interactive Streamlit dashboard.
- Allow users to configure electricity tariffs.
- Add uncertainty estimates to forecasts.
- Evaluate the system on additional public energy datasets.

## 8. Conclusion

The architecture separates data processing, feature engineering, model training, forecasting, and visualization into individual modules. This makes the project easier to test, maintain, document, and extend.

The current implementation demonstrates an end-to-end educational forecasting workflow. Additional validation and time-series quality checks are required before using it for operational decisions.
