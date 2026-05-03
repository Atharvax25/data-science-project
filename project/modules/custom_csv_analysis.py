"""Custom CSV loading and analysis for user-provided datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_custom_csv(file_path: str) -> pd.DataFrame:
    """Load a user-provided CSV file using an absolute or relative path."""
    path = Path(file_path.strip().strip('"')).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Please provide a valid .csv file.")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The selected CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The selected CSV file could not be parsed correctly.") from exc


def show_dataset_overview(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("\n--- Custom CSV Dataset Overview ---")
    print("Rows and columns:", df.shape)
    print("\nColumn names:")
    print(list(df.columns))
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())


def show_numeric_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    print("\n--- Numeric Statistics ---")
    if numeric_df.empty:
        print("No numeric columns found.")
        return

    print(numeric_df.describe())
    print("\nMean:")
    print(numeric_df.mean())
    print("\nMedian:")
    print(numeric_df.median())
    print("\nVariance:")
    print(numeric_df.var())
    print("\nStandard deviation:")
    print(numeric_df.std())


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and text values with mode."""
    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
        else:
            mode_values = cleaned_df[column].mode()
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    print("\nMissing values handled successfully.")
    print(cleaned_df.isnull().sum())
    return cleaned_df


def filter_rows_interactive(df: pd.DataFrame) -> None:
    """Filter rows using user input."""
    print("\nAvailable columns:")
    print(list(df.columns))

    column = input("Enter column name to filter: ").strip()
    if column not in df.columns:
        print("Column not found.")
        return

    operator = input("Enter operator (=, !=, >, <, >=, <=): ").strip()
    value = input("Enter value: ").strip()

    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            value = float(value)

        if operator == "=":
            filtered_df = df[df[column] == value]
        elif operator == "!=":
            filtered_df = df[df[column] != value]
        elif operator == ">":
            filtered_df = df[df[column] > value]
        elif operator == "<":
            filtered_df = df[df[column] < value]
        elif operator == ">=":
            filtered_df = df[df[column] >= value]
        elif operator == "<=":
            filtered_df = df[df[column] <= value]
        else:
            print("Invalid operator.")
            return

        print("\nFiltered rows:")
        print(filtered_df.head(20))
        print(f"\nTotal filtered rows: {len(filtered_df)}")
    except ValueError:
        print("Invalid value for this column.")
    except TypeError:
        print("This operation is not suitable for the selected column.")


def groupby_aggregation_interactive(df: pd.DataFrame) -> None:
    """Perform groupby aggregation using user input."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    print("\nAvailable columns:")
    print(list(df.columns))
    print("\nNumeric columns:")
    print(numeric_columns)

    group_column = input("Enter column to group by: ").strip()
    value_column = input("Enter numeric column to aggregate: ").strip()

    if group_column not in df.columns:
        print("Group column not found.")
        return

    if value_column not in numeric_columns:
        print("Please select a valid numeric column.")
        return

    result = df.groupby(group_column)[value_column].agg(["sum", "mean", "min", "max", "count"])
    print("\nGroupby aggregation result:")
    print(result)


def plot_histogram_interactive(df: pd.DataFrame) -> None:
    """Plot histogram for a selected numeric column."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        print("No numeric columns available for histogram.")
        return

    print("\nNumeric columns:")
    print(numeric_columns)
    column = input("Enter numeric column for histogram: ").strip()

    if column not in numeric_columns:
        print("Invalid numeric column.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True, color="steelblue")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        print("At least two numeric columns are needed for correlation analysis.")
        return

    print("\nCorrelation matrix:")
    print(numeric_df.corr())

    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def show_custom_csv_menu() -> None:
    """Display menu for custom CSV operations."""
    print("\n--- Custom CSV Operations ---")
    print("1. Dataset overview")
    print("2. Numeric statistics")
    print("3. Handle missing values")
    print("4. Filter rows")
    print("5. Groupby aggregation")
    print("6. Plot histogram")
    print("7. Correlation analysis")
    print("8. Run all automatic operations")
    print("0. Back to main menu")


def run_automatic_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Run automatic operations that do not need extra user input."""
    show_dataset_overview(df)
    show_numeric_statistics(df)
    cleaned_df = clean_missing_values(df)
    plot_correlation_heatmap(cleaned_df)
    return cleaned_df


def run_custom_csv_analysis() -> None:
    """Load a custom CSV and let the user choose operations."""
    file_path = input("Enter CSV file path: ").strip()
    df = load_custom_csv(file_path)

    while True:
        show_custom_csv_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            show_dataset_overview(df)
        elif choice == "2":
            show_numeric_statistics(df)
        elif choice == "3":
            df = clean_missing_values(df)
        elif choice == "4":
            filter_rows_interactive(df)
        elif choice == "5":
            groupby_aggregation_interactive(df)
        elif choice == "6":
            plot_histogram_interactive(df)
        elif choice == "7":
            plot_correlation_heatmap(df)
        elif choice == "8":
            df = run_automatic_operations(df)
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please enter a number from 0 to 8.")
