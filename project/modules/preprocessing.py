"""Data loading, preprocessing, and Titanic data analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(file_name: str) -> pd.DataFrame:
    """Load a CSV dataset from the data folder."""
    file_path = DATA_DIR / file_name
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {file_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {file_path}") from exc


def basic_data_analysis(df: pd.DataFrame) -> None:
    """Perform filtering, groupby operations, and aggregations."""
    print("\n--- Basic Data Analysis With Pandas ---")
    print("First five rows:")
    print(df.head())

    adults = df[df["Age"] >= 18]
    print("\nFiltered rows where Age >= 18:")
    print(adults[["Name", "Sex", "Age", "Fare"]])

    print("\nAverage survival by gender:")
    print(df.groupby("Sex")["Survived"].mean())

    print("\nFare aggregations by passenger class:")
    print(df.groupby("Pclass")["Fare"].agg(["sum", "mean", "min", "max"]))


def handle_titanic_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing Titanic values using simple beginner-friendly methods."""
    cleaned_df = df.copy()
    cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())
    cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])
    cleaned_df["Fare"] = cleaned_df["Fare"].fillna(cleaned_df["Fare"].median())
    return cleaned_df


def perform_titanic_eda(df: pd.DataFrame) -> None:
    """Print simple exploratory data analysis details."""
    numeric_df = df.select_dtypes(include="number")

    print("\n--- Titanic Exploratory Data Analysis ---")
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(numeric_df.describe())
    print("\nSurvival count:")
    print(df["Survived"].value_counts())


def plot_survival_vs_gender(df: pd.DataFrame) -> None:
    """Plot Titanic survival by gender."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Sex", hue="Survived", palette="Set2")
    plt.title("Titanic Survival vs Gender")
    plt.xlabel("Gender")
    plt.ylabel("Passenger Count")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()


def plot_survival_vs_class(df: pd.DataFrame) -> None:
    """Plot Titanic survival by passenger class."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1")
    plt.title("Titanic Survival vs Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Passenger Count")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()


def run_titanic_analysis() -> None:
    """Run complete Titanic data analysis."""
    titanic_df = load_dataset("titanic.csv")
    basic_data_analysis(titanic_df)
    cleaned_df = handle_titanic_missing_values(titanic_df)
    perform_titanic_eda(cleaned_df)
    plot_survival_vs_gender(cleaned_df)
    plot_survival_vs_class(cleaned_df)
