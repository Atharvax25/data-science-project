"""Application entry point for the Smart Data Analytics and Prediction System."""

from __future__ import annotations

from modules.classification import run_spam_detection
from modules.custom_csv_analysis import run_custom_csv_analysis
from modules.linear_algebra import run_linear_algebra_demo
from modules.preprocessing import run_titanic_analysis
from modules.recommendation import run_recommendation_system
from modules.regression import run_regression_models
from modules.statistics import run_statistics_demo
from web_app import main as run_ui


def show_menu() -> None:
    """Display the main menu."""
    print("\n==============================================")
    print("Smart Data Analytics and Prediction System")
    print("==============================================")
    print("1. Linear Algebra")
    print("2. Probability and Statistics")
    print("3. Titanic Data Analysis")
    print("4. Machine Learning Models")
    print("5. Run Full Project")
    print("6. Analyze Your Own CSV File")
    print("0. Exit")


def run_ml_models() -> None:
    """Run all machine learning modules."""
    run_regression_models()
    run_spam_detection()
    run_recommendation_system()


def run_full_project() -> None:
    """Run every project module."""
    run_linear_algebra_demo()
    run_statistics_demo()
    run_titanic_analysis()
    run_ml_models()


def cli_main() -> None:
    """Start the menu-driven CLI."""
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()

        try:
            if choice == "1":
                run_linear_algebra_demo()
            elif choice == "2":
                run_statistics_demo()
            elif choice == "3":
                run_titanic_analysis()
            elif choice == "4":
                run_ml_models()
            elif choice == "5":
                run_full_project()
            elif choice == "6":
                run_custom_csv_analysis()
            elif choice == "0":
                print("Thank you for using the project. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number from 0 to 6.")
        except Exception as error:
            print(f"An error occurred: {error}")


def main() -> None:
    """Launch the professional browser UI by default."""
    run_ui()


if __name__ == "__main__":
    main()
