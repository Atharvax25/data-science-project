"""Regression models for house price and Uber fare prediction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from modules.preprocessing import load_dataset


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate RMSE in a scikit-learn version friendly way."""
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)


def run_house_price_prediction() -> None:
    """Train a linear regression model for house price prediction."""
    df = load_dataset("house_prices.csv")
    x = df[["area", "bedrooms", "age"]]
    y = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    sample_house = pd.DataFrame([[2200, 3, 5]], columns=["area", "bedrooms", "age"])
    predicted_price = model.predict(sample_house)[0]

    print("\n--- Linear Regression: House Price Prediction ---")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"RMSE: {calculate_rmse(y_test, predictions):.2f}")
    print(f"Predicted price for area=2200, bedrooms=3, age=5: {predicted_price:.2f}")


def preprocess_uber_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Uber fare data and remove outliers."""
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df[cleaned_df["fare_amount"] > 0]
    cleaned_df = cleaned_df[cleaned_df["distance_km"] > 0]

    q1 = cleaned_df["fare_amount"].quantile(0.25)
    q3 = cleaned_df["fare_amount"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return cleaned_df[(cleaned_df["fare_amount"] >= lower_bound) & (cleaned_df["fare_amount"] <= upper_bound)]


def plot_uber_correlation(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for Uber data."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Uber Fare Correlation Analysis")
    plt.tight_layout()
    plt.show()


def evaluate_regression_model(model, x_test, y_test, model_name: str) -> None:
    """Evaluate and print regression model scores."""
    predictions = model.predict(x_test)
    print(f"\n{model_name}")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"RMSE: {calculate_rmse(y_test, predictions):.2f}")


def run_uber_fare_prediction() -> None:
    """Train Linear Regression and Random Forest models for Uber fare prediction."""
    df = load_dataset("uber_fares.csv")
    cleaned_df = preprocess_uber_data(df)

    print("\n--- Uber Fare Prediction ---")
    print("Rows before preprocessing:", len(df))
    print("Rows after preprocessing:", len(cleaned_df))
    print("\nCorrelation matrix:")
    print(cleaned_df.corr(numeric_only=True))

    features = ["distance_km", "passenger_count", "hour", "day_of_week"]
    x = cleaned_df[features]
    y = cleaned_df["fare_amount"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    evaluate_regression_model(linear_model, x_test, y_test, "Linear Regression")

    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_model.fit(x_train, y_train)
    evaluate_regression_model(forest_model, x_test, y_test, "Random Forest Regressor")

    sample_ride = pd.DataFrame([[6.0, 2, 18, 5]], columns=features)
    print(f"Sample ride predicted fare: {forest_model.predict(sample_ride)[0]:.2f}")
    plot_uber_correlation(cleaned_df)


def run_regression_models() -> None:
    """Run all regression models."""
    run_house_price_prediction()
    run_uber_fare_prediction()
