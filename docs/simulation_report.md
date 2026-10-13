# Simulation and Model Evaluation Report

## AI-Powered Energy Consumption Forecasting System

## 1. Executive Summary

This project demonstrates an end-to-end machine learning workflow for forecasting household electricity consumption using a publicly available historical dataset.

The pipeline loads and prepares electricity readings, generates forecasting features, trains and evaluates machine learning models, forecasts the next 24 hourly periods, generates visualizations, and estimates illustrative electricity costs.

The implementation completed an end-to-end run successfully. Random Forest produced the lowest RMSE among the evaluated models in the initial chronological holdout experiment.

This report documents the experimental results and assumptions. The results are preliminary and should not be interpreted as evidence of production readiness.

## 2. Dataset Description

**Dataset:** Individual Household Electric Power Consumption  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains minute-level household electricity measurements collected in France from December 2006 to November 2010.

The project uses the `Global_active_power` column to estimate hourly energy consumption.

### Data preparation results

| Measurement | Result |
|---|---:|
| Original observations | 2,075,259 |
| Valid hourly records | 34,173 |
| Usable records after feature engineering | 34,005 |
| Engineered features | 8 |
| Missing values in processed output | 0 |

The source readings are processed into hourly energy estimates. Valid minute-level active power readings are averaged within each hour. The hourly mean power in kW corresponds numerically to hourly energy in kWh when the interval is one hour.

The preprocessing stage handles missing readings and interpolates eligible short gaps.

Although the processed output contains no missing values, this does not guarantee that every consecutive row represents consecutive hours. Explicit handling of missing hourly timestamps remains an important improvement.

## 3. Feature Engineering

Eight features are created for forecasting:

| Feature | Description |
|---|---|
| `hour` | Hour of day |
| `day_of_week` | Day of the week |
| `day_of_year` | Day of the year |
| `is_weekend` | Weekend indicator |
| `lag_1` | Previous available observation |
| `lag_24` | Observation 24 rows earlier |
| `lag_168` | Observation 168 rows earlier |
| `rolling_mean_24` | Mean of the previous 24 available observations |

The current implementation uses row-based shifts and rolling windows. If hourly timestamps contain gaps, these windows may not correspond to the intended elapsed time. A future version should reindex the data to an explicit hourly timeline and handle missing periods before creating features.

## 4. Experimental Method

Three forecasting approaches were compared:

1. Previous-day baseline
2. Linear Regression
3. Random Forest Regressor

The data was split chronologically rather than randomly shuffled.

| Dataset portion | Records |
|---|---:|
| Training | 27,204 |
| Testing | 6,801 |

The evaluated metrics were Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

- **MAE:** Average absolute prediction error.
- **RMSE:** Prediction error metric that gives greater weight to larger errors.
- **R²:** Measures model performance relative to predicting the mean of the target values.

## 5. Model Evaluation Results

The initial training run produced these results:

| Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---:|---:|---:|
| Random Forest | 0.3370 | 0.4891 | 0.5667 |
| Linear Regression | 0.3729 | 0.5227 | 0.5051 |
| Previous-day baseline | 0.5356 | 0.7903 | -0.1315 |

### Interpretation

Random Forest achieved the lowest RMSE of the three evaluated approaches. Its MAE was also lower than the Linear Regression and previous-day baseline results.

The Random Forest model was selected for forecasting by the implemented training procedure.

These results are a useful initial benchmark, not a guarantee of future prediction accuracy.

### Evaluation caveat

The current training procedure uses test-set performance when selecting the best machine-learning model. Therefore, the reported best-model score is not a completely independent final estimate of generalization performance.

For a stronger experiment, the data should be divided into chronological training, validation, and final test periods. Model selection should use validation data, leaving the final test period untouched until the end.

Walk-forward time-series evaluation should also be considered.

## 6. Twenty-Four-Hour Forecast

The forecasting module generated 24 hourly predictions using the selected Random Forest model.

| Forecast metric | Result |
|---|---:|
| Forecast periods | 24 hours |
| Total predicted energy | 39.07 kWh |
| Average predicted hourly energy | 1.63 kWh |
| Peak predicted hourly energy | 3.72 kWh |

The forecast output is saved to:

`outputs/next_24_hours.csv`

The source dataset ends at 20:00 on November 26, 2010. The displayed forecast therefore starts at 21:00 on November 26, 2010, and ends at 20:00 on November 27, 2010.

This is a forecast based on historical data, not a prediction of current household consumption.

The forecasting process is recursive: predicted values are added to the available history before producing subsequent predictions. Errors can accumulate over the 24-hour horizon.

## 7. Illustrative Electricity-Cost Simulation

The project applies an assumed flat electricity tariff of ₹8 per kWh to the predicted hourly energy.

| Simulation item | Result |
|---|---:|
| Predicted energy over 24 hours | 39.07 kWh |
| Assumed tariff | ₹8.00/kWh |
| Estimated 24-hour cost | ₹312.57 |

The cost calculation is performed for each forecast period and the hourly estimates are summed.

This simulation is illustrative only. It is not an actual electricity bill and does not include tariff slabs, taxes, fixed charges, time-of-use pricing, or utility-specific billing rules.

The output is saved to:

`outputs/business_simulation.csv`

## 8. Generated Project Artifacts

### Model artifacts

- `models/best_model.joblib`
- `outputs/metrics.csv`
- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`

### Forecast artifacts

- `outputs/next_24_hours.csv`

### Simulation artifact

- `outputs/business_simulation.csv`

### Visualizations

- `images/consumption_trend.png`
- `images/actual_vs_predicted.png`
- `images/model_comparison.png`
- `images/next_24_hours_forecast.png`

These files support the project's analysis and can be reviewed when presenting the work on GitHub.

## 9. Reproducing the Experiment

Activate the project's Python virtual environment and run the following commands from the project root:

```cmd
python main.py prepare
python main.py train
python main.py forecast
python main.py visualize
```

To run every stage in sequence:

```cmd
python main.py all
```

The verified end-to-end run completed data preparation, model training, forecasting, visualization, and illustrative cost simulation successfully.

## 10. Limitations

The current experiment has several limitations:

- The dataset represents one household and is not an industrial energy dataset.
- The observations end in November 2010 and are not live data.
- Missing hourly periods need more explicit treatment.
- The model-selection procedure uses test performance and should be redesigned.
- Forecasting recursively can propagate prediction errors.
- The cost calculation uses an assumed constant tariff.
- No uncertainty intervals are currently produced.

Consequently, the model should be treated as an educational forecasting demonstration rather than an operational decision system.

## 11. Future Improvements

Recommended improvements include:

1. Reindex hourly data to a complete timeline and handle gaps explicitly.
2. Create separate chronological training, validation, and final test periods.
3. Add walk-forward validation.
4. Compare Random Forest with seasonal-naive and other forecasting baselines.
5. Add forecast uncertainty estimates.
6. Support configurable electricity tariffs.
7. Build an interactive Streamlit dashboard.
8. Evaluate performance on additional energy datasets.

## 12. Conclusion

The project demonstrates a working pipeline for data preparation, feature engineering, model comparison, 24-hour forecasting, visualization, and cost simulation.

The initial experiment suggests that Random Forest performed better than the other evaluated approaches on the reported chronological holdout split. Stronger time-series validation and explicit missing-hour handling are needed before drawing more robust conclusions about model performance.

The result is a useful educational and portfolio project that can be extended with improved validation, interactive visualization, and more realistic forecasting scenarios.
