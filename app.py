
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Energy Intelligence | Data Science",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# --------------------------------------------------
# PROFESSIONAL UI THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #e5edf8;
    }

    [data-testid="stSidebar"] {
        background: #101a2d;
        border-right: 1px solid #23314a;
    }

    [data-testid="stSidebar"] * {
        color: #e5edf8;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 26px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #142640, #102d37);
        border: 1px solid #294657;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #f5f9ff;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .hero p {
        color: #a9bfd3;
        margin-bottom: 0;
        line-height: 1.7;
    }

    .eyebrow {
        color: #5eead4;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .metric-card {
        background: linear-gradient(145deg, #142138, #111b2d);
        border: 1px solid #263750;
        border-radius: 15px;
        padding: 20px;
        min-height: 132px;
    }

    .metric-label {
        color: #9db0c9;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #f1f7ff;
        font-size: 1.7rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #5eead4;
        font-size: 0.77rem;
        margin-top: 8px;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f7ff;
        margin: 20px 0 5px 0;
    }

    .section-subtitle {
        color: #92a5bd;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        border: 1px solid #25344a;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: #142138;
        border: 1px solid #263750;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9db0c9;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f7ff;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #2b827e;
    }

    .footer {
        text-align: center;
        color: #8293a9;
        font-size: 0.8rem;
        padding-top: 30px;
        border-top: 1px solid #25344a;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_csv(path):
    """Load a project CSV if it exists."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def first_existing(paths):
    """Return the first available file from a list."""
    for path in paths:
        if path.exists():
            return path
    return None


def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#25344a", zeroline=False)
    return fig


def numeric_columns(df):
    return list(df.select_dtypes(include="number").columns)


def format_number(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Energy Intelligence")
    st.caption("DATA SCIENCE DASHBOARD")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Energy Analytics",
            "Forecast",
            "Model Performance",
            "Data Explorer",
        ],
    )

    st.divider()
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload a compatible dataset to explore its columns and trends.",
    )

    st.caption("Historical data and project outputs are loaded locally when available.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
default_path = first_existing(
    [
        PROCESSED_DIR / "hourly_energy.csv",
        BASE_DIR / "data" / "processed_energy.csv",
    ]
)

df = None
data_source = "No dataset loaded"

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        data_source = f"Uploaded: {uploaded_file.name}"
    except (ValueError, pd.errors.ParserError, OSError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
elif default_path is not None:
    df = load_csv(default_path)
    data_source = str(default_path.relative_to(BASE_DIR))

metrics_df = load_csv(OUTPUTS_DIR / "metrics.csv")
comparison_df = load_csv(OUTPUTS_DIR / "model_comparison.csv")
predictions_df = load_csv(OUTPUTS_DIR / "test_predictions.csv")
forecast_df = load_csv(OUTPUTS_DIR / "next_24_hours.csv")

if df is not None:
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        st.warning("The selected CSV has no rows.")
        st.stop()

    date_col = find_column(
        df,
        ["timestamp", "datetime", "date", "time", "Datetime", "Date"],
    )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    energy_col = find_column(
        df,
        [
            "Global_active_power",
            "global_active_power",
            "energy_kwh",
            "energy",
            "consumption",
            "power",
            "predicted_energy_kwh",
        ],
    )

    # Avoid treating an arbitrary identifier as energy.
    if energy_col is None:
        candidates = [
            c for c in numeric_columns(df)
            if c.lower() not in {
                "id", "hour", "day", "day_of_week", "day_of_year",
                "is_weekend", "temperature", "pressure", "windspeed",
                "var1", "var2",
            }
        ]
        if candidates:
            energy_col = candidates[0]

    if energy_col:
        df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">DATA SCIENCE • PREDICTIVE ANALYTICS</div>
        <h1>Energy Consumption Intelligence</h1>
        <p>
            Explore historical electricity usage, understand consumption
            patterns, and review energy forecasts and model performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Data source: {data_source}")

if df is None:
    st.info(
        "No default dataset was found. Upload a CSV from the sidebar, "
        "or run the project's data preparation pipeline first."
    )
    st.code("python main.py prepare", language="bash")
    st.caption(
        "If your project uses a different command, use the command "
        "documented in your README."
    )
    st.stop()

if energy_col is None:
    st.warning(
        "No suitable energy column was detected. You can still inspect "
        "the data in Data Explorer."
    )

# --------------------------------------------------
# METRICS
# --------------------------------------------------
energy_values = (
    df[energy_col].dropna()
    if energy_col is not None
    else pd.Series(dtype=float)
)

total_energy = energy_values.sum() if not energy_values.empty else None
average_energy = energy_values.mean() if not energy_values.empty else None
peak_energy = energy_values.max() if not energy_values.empty else None

rmse_col = find_column(metrics_df, ["RMSE", "rmse"]) if metrics_df is not None else None
mae_col = find_column(metrics_df, ["MAE", "mae"]) if metrics_df is not None else None
r2_col = find_column(metrics_df, ["R2", "R²", "r2", "r2_score"]) if metrics_df is not None else None

rmse_value = metrics_df[rmse_col].iloc[0] if rmse_col and not metrics_df.empty else None
mae_value = metrics_df[mae_col].iloc[0] if mae_col and not metrics_df.empty else None
r2_value = metrics_df[r2_col].iloc[0] if r2_col and not metrics_df.empty else None

if page == "Overview":
    st.markdown('<div class="section-title">Key indicators</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A quick summary of the selected dataset.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL ENERGY",
            f"{format_number(total_energy)}" if total_energy is not None else "N/A",
            f"Column: {energy_col}" if energy_col else "Energy column not detected",
        )
    with c2:
        metric_card(
            "AVERAGE VALUE",
            format_number(average_energy) if average_energy is not None else "N/A",
            "Mean of available numeric values",
        )
    with c3:
        metric_card(
            "PEAK VALUE",
            format_number(peak_energy) if peak_energy is not None else "N/A",
            "Maximum recorded value",
        )
    with c4:
        metric_card(
            "DATA RECORDS",
            f"{len(df):,}",
            f"{len(df.columns)} columns",
        )

    st.markdown('<div class="section-title">Consumption overview</div>', unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        if energy_col and not energy_values.empty:
            chart_df = df.copy()

            if date_col and chart_df[date_col].notna().any():
                chart_df = chart_df.dropna(subset=[date_col])
                chart_df = chart_df.sort_values(date_col)
                x_col = date_col
            else:
                chart_df = chart_df.reset_index(drop=True)
                x_col = chart_df.index.name or "Record"

            if len(chart_df) > 3000:
                chart_df = chart_df.tail(3000)

            fig = px.line(
                chart_df,
                x=x_col,
                y=energy_col,
                title="Energy consumption over time",
                labels={energy_col: energy_col, x_col: "Time / Record"},
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            st.info("Upload a dataset containing a numeric energy column to see this chart.")

    with right:
        st.markdown("#### Dataset profile")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", f"{len(df.columns):,}")
        st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        if date_col:
            valid_dates = df[date_col].dropna()
            if not valid_dates.empty:
                st.caption(
                    f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} "
                    f"to {valid_dates.max().strftime('%Y-%m-%d')}"
                )

    st.markdown('<div class="section-title">Model snapshot</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("MAE", format_number(mae_value) if mae_value is not None else "Not available")
    with m2:
        st.metric("RMSE", format_number(rmse_value) if rmse_value is not None else "Not available")
    with m3:
        st.metric("R² score", format_number(r2_value) if r2_value is not None else "Not available")

    if metrics_df is None:
        st.caption("Model metrics will appear after the project training pipeline generates outputs/metrics.csv.")

elif page == "Energy Analytics":
    st.markdown("## Energy Analytics")
    st.markdown(
        '<div class="section-subtitle">Explore numeric columns and consumption patterns.</div>',
        unsafe_allow_html=True,
    )

    numeric_cols = numeric_columns(df)

    if not numeric_cols:
        st.warning("No numeric columns are available for analysis.")
    else:
        selected_col = st.selectbox(
            "Select a metric",
            numeric_cols,
            index=numeric_cols.index(energy_col) if energy_col in numeric_cols else 0,
        )

        valid_values = pd.to_numeric(df[selected_col], errors="coerce").dropna()
        a, b, c = st.columns(3)
        a.metric("Mean", format_number(valid_values.mean()) if not valid_values.empty else "N/A")
        b.metric("Median", format_number(valid_values.median()) if not valid_values.empty else "N/A")
        c.metric("Standard deviation", format_number(valid_values.std()) if not valid_values.empty else "N/A")

        chart_data = df.copy()
        if date_col and chart_data[date_col].notna().any():
            chart_data = chart_data.dropna(subset=[date_col]).sort_values(date_col)
            x = date_col
        else:
            chart_data = chart_data.reset_index(drop=True)
            chart_data["_record_number"] = range(1, len(chart_data) + 1)
            x = "_record_number"

        if len(chart_data) > 5000:
            chart_data = chart_data.tail(5000)

        fig = px.line(
            chart_data,
            x=x,
            y=selected_col,
            title=f"{selected_col} trend",
            labels={x: "Time / Record", selected_col: selected_col},
        )
        st.plotly_chart(style_chart(fig, 420), use_container_width=True)

        left, right = st.columns(2)

        with left:
            hist = px.histogram(
                chart_data,
                x=selected_col,
                nbins=35,
                title=f"Distribution of {selected_col}",
            )
            st.plotly_chart(style_chart(hist), use_container_width=True)

        with right:
            if len(numeric_cols) >= 2:
                y_col = st.selectbox(
                    "Compare with",
                    [c for c in numeric_cols if c != selected_col],
                )
                scatter = px.scatter(
                    chart_data,
                    x=selected_col,
                    y=y_col,
                    title=f"{selected_col} vs {y_col}",
                    opacity=0.65,
                )
                st.plotly_chart(style_chart(scatter), use_container_width=True)
            else:
                st.info("Upload at least two numeric columns to compare variables.")

elif page == "Forecast":
    st.markdown("## Energy Forecast")
    st.markdown(
        '<div class="section-subtitle">Forecast results generated by your project pipeline.</div>',
        unsafe_allow_html=True,
    )

    if forecast_df is None or forecast_df.empty:
        st.warning("Forecast results were not found.")
        st.code("python main.py forecast", language="bash")
        st.caption(
            "Run the forecast command from your project directory, then refresh the dashboard."
        )
    else:
        st.success(f"Loaded {len(forecast_df)} forecast records.")

        forecast_time = find_column(
            forecast_df,
            ["timestamp", "datetime", "date", "time"],
        )
        forecast_value = find_column(
            forecast_df,
            [
                "predicted_energy_kwh",
                "forecast",
                "predicted",
                "prediction",
                "energy",
                "Global_active_power",
            ],
        )

        if forecast_time:
            forecast_df[forecast_time] = pd.to_datetime(
                forecast_df[forecast_time], errors="coerce"
            )

        if forecast_value is None:
            numeric = numeric_columns(forecast_df)
            if numeric:
                forecast_value = numeric[-1]

        if forecast_value:
            vals = pd.to_numeric(forecast_df[forecast_value], errors="coerce").dropna()

            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast total", format_number(vals.sum()) if not vals.empty else "N/A")
            c2.metric("Average forecast", format_number(vals.mean()) if not vals.empty else "N/A")
            c3.metric("Peak forecast", format_number(vals.max()) if not vals.empty else "N/A")

            if forecast_time:
                plot_df = forecast_df.dropna(subset=[forecast_time]).sort_values(forecast_time)
                x_col = forecast_time
            else:
                plot_df = forecast_df.copy().reset_index(drop=True)
                plot_df["Forecast step"] = range(1, len(plot_df) + 1)
                x_col = "Forecast step"

            fig = px.line(
                plot_df,
                x=x_col,
                y=forecast_value,
                markers=True,
                title="Forecast for the next available time steps",
                labels={x_col: "Time", forecast_value: "Forecast value"},
            )
            st.plotly_chart(style_chart(fig, 430), use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="energy_forecast.csv",
                mime="text/csv",
            )
        else:
            st.warning("No numeric forecast column was detected.")
            st.dataframe(forecast_df.head(20), use_container_width=True)

elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.markdown(
        '<div class="section-subtitle">Review model metrics and actual-versus-predicted results.</div>',
        unsafe_allow_html=True,
    )

    if comparison_df is not None and not comparison_df.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        metric_name = find_column(comparison_df, ["model", "Model", "algorithm"])
        metric_cols = [
            c for c in comparison_df.columns
            if c.lower() in {"mae", "rmse", "r2", "r²", "r2_score", "failure_f1"}
        ]

        if metric_name and metric_cols:
            chosen_metric = st.selectbox("Metric to compare", metric_cols)
            plot_df = comparison_df.copy()
            plot_df[chosen_metric] = pd.to_numeric(plot_df[chosen_metric], errors="coerce")
            fig = px.bar(
                plot_df,
                x=metric_name,
                y=chosen_metric,
                title=f"Models compared by {chosen_metric}",
                color=metric_name,
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)
    elif metrics_df is not None and not metrics_df.empty:
        st.markdown("### Available evaluation metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No model metrics were found. Run the training pipeline to generate "
            "outputs/metrics.csv and outputs/model_comparison.csv."
        )
        st.code("python main.py train", language="bash")

    if predictions_df is not None and not predictions_df.empty:
        st.markdown("### Actual vs. predicted")
        st.dataframe(predictions_df.head(20), use_container_width=True, hide_index=True)

        actual_col = find_column(
            predictions_df, ["actual", "y_true", "actual_energy", "actual_kwh"]
        )
        predicted_col = find_column(
            predictions_df, ["predicted", "y_pred", "prediction", "predicted_energy"]
        )

        if actual_col and predicted_col:
            plot_df = predictions_df[[actual_col, predicted_col]].copy()
            plot_df[actual_col] = pd.to_numeric(plot_df[actual_col], errors="coerce")
            plot_df[predicted_col] = pd.to_numeric(plot_df[predicted_col], errors="coerce")
            plot_df = plot_df.dropna()

            if not plot_df.empty:
                plot_df = plot_df.head(1000).reset_index(drop=True)
                plot_df["Test record"] = range(1, len(plot_df) + 1)

                fig = px.line(
                    plot_df,
                    x="Test record",
                    y=[actual_col, predicted_col],
                    title="Actual vs. predicted energy",
                    labels={"value": "Energy", "variable": "Series"},
                )
                st.plotly_chart(style_chart(fig), use_container_width=True)
    else:
        st.caption("Prediction-level data is not available yet.")

elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    st.markdown(
        '<div class="section-subtitle">Filter, inspect, and download the selected dataset.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Filters")
    filtered_df = df.copy()

    if date_col and filtered_df[date_col].notna().any():
        valid_dates = filtered_df[date_col].dropna()
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                filtered_df[date_col].dt.date.between(start_date, end_date)
            ]

    search = st.text_input("Search column names")
    if search:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: row.astype(str).str.contains(
                    search, case=False, na=False
                ).any(),
                axis=1,
            )
        ]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
    st.dataframe(filtered_df.head(5000), use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="energy_data_export.csv",
        mime="text/csv",
    )

st.markdown(
    """
    <div class="footer">
        ENERGY INTELLIGENCE · DATA SCIENCE PORTFOLIO PROJECT<br>
        Built with Python, Streamlit, Pandas and Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
