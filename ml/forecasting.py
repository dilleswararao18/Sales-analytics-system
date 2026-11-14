
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

FEATURES = [
    "month",
    "year",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean"
]


def prepare_monthly_data(sales):

    if not sales:
        return pd.DataFrame(
            columns=["Date", "Revenue"]
        )

    df = pd.DataFrame([
        {
            "Date": sale.sale_date,
            "Revenue": sale.revenue
        }
        for sale in sales
    ])

    df["Date"] = pd.to_datetime(df["Date"])

    monthly = (
        df.set_index("Date")["Revenue"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    dates = pd.date_range(
        start=monthly["Date"].min(),
        end=monthly["Date"].max(),
        freq="MS"
    )

    monthly = (
        monthly.set_index("Date")
        .reindex(dates, fill_value=0)
        .rename_axis("Date")
        .reset_index()
    )

    return monthly


def create_features(monthly):

    data = monthly.copy()

    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    data["lag_1"] = data["Revenue"].shift(1)
    data["lag_2"] = data["Revenue"].shift(2)
    data["lag_3"] = data["Revenue"].shift(3)

    data["rolling_mean"] = (
        data["Revenue"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    return data.dropna().reset_index(drop=True)



def plot_to_html(fig):

    fig.update_layout(
        height=420,
        autosize=True,
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=60
        ),
        template="plotly_white"
    )

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False
        }
    )



def train_forecast_model(sales, forecast_periods=3):

    monthly = prepare_monthly_data(sales)

    if len(monthly) < 12:

        return {
            "success": False,
            "message": (
                f"Only {len(monthly)} months of data available. "
                "At least 12 months are required. "
                "24 or more months are recommended."
            )
        }

    data = create_features(monthly)

    if len(data) < 8:

        return {
            "success": False,
            "message": (
                "Not enough historical observations "
                "to train and evaluate the model."
            )
        }

    # Chronological train-test split
    test_size = max(
        2,
        int(np.ceil(len(data) * 0.20))
    )

    X_train = data[FEATURES].iloc[:-test_size]
    X_test = data[FEATURES].iloc[-test_size:]

    y_train = data["Revenue"].iloc[:-test_size]
    y_test = data["Revenue"].iloc[-test_size:]

    # Evaluation model
    evaluation_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )

    evaluation_model.fit(X_train, y_train)

    predictions = np.maximum(
        0,
        evaluation_model.predict(X_test)
    )

    # Model evaluation
    mae = float(
        mean_absolute_error(y_test, predictions)
    )

    rmse = float(
        np.sqrt(mean_squared_error(y_test, predictions))
    )

    r2 = float(
        r2_score(y_test, predictions)
    )

    # Actual vs Predicted chart
    comparison = pd.DataFrame({
        "Date": data["Date"].iloc[-test_size:],
        "Actual Revenue": y_test.to_numpy(),
        "Predicted Revenue": predictions
    })

    comparison_long = comparison.melt(
        id_vars="Date",
        var_name="Series",
        value_name="Revenue"
    )

    comparison_fig = px.line(
        comparison_long,
        x="Date",
        y="Revenue",
        color="Series",
        markers=True,
        title="Actual vs Predicted Monthly Revenue"
    )

    comparison_fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Revenue"
    )

    # Final model trained on all available data
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(
        data[FEATURES],
        data["Revenue"]
    )

    # Generate future predictions
    history = monthly["Revenue"].astype(float).tolist()

    future_dates = pd.date_range(
        start=monthly["Date"].iloc[-1]
        + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq="MS"
    )

    forecast_values = []

    for future_date in future_dates:

        row = pd.DataFrame([{
            "month": future_date.month,
            "year": future_date.year,
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3],
            "rolling_mean": float(
                np.mean(history[-3:])
            )
        }])

        prediction = max(
            0.0,
            float(model.predict(row[FEATURES])[0])
        )

        forecast_values.append(prediction)

        history.append(prediction)

    # Historical and future forecast chart
    historical = monthly.copy()
    historical["Type"] = "Historical"

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Revenue": forecast_values,
        "Type": "Forecast"
    })

    combined = pd.concat(
        [historical, forecast_df],
        ignore_index=True
    )

    forecast_fig = px.line(
        combined,
        x="Date",
        y="Revenue",
        color="Type",
        markers=True,
        title="Historical Sales vs Future Forecast"
    )

    forecast_fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Revenue"
    )

    # Standardized forecast records for dashboard
    forecast_rows = []

    for date, value in zip(future_dates, forecast_values):

        forecast_rows.append({
            "month": date.strftime("%B %Y"),
            "predicted_sales": float(value)
        })

    return {
        "success": True,

        "message": "Sales forecasting completed successfully.",

        "chart": plot_to_html(forecast_fig),

        "comparison_chart": plot_to_html(comparison_fig),

        "mae": mae,
        "rmse": rmse,
        "r2": r2,

        # Dashboard forecast table
        "forecast": forecast_rows,

        # Keep this alias for compatibility
        "forecast_rows": forecast_rows,

        "test_rows": comparison.to_dict("records")
    }
