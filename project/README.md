# Smart Data Analytics and Prediction System

A beginner-friendly end-to-end Python project for learning linear algebra, probability, statistics, data analysis, Titanic EDA, machine learning models, and custom CSV analysis.

## Project Structure

```text
project/
|-- data/
|   |-- emails.csv
|   |-- house_prices.csv
|   |-- movie_ratings.csv
|   |-- titanic.csv
|   |-- uber_fares.csv
|-- modules/
|   |-- __init__.py
|   |-- classification.py
|   |-- custom_csv_analysis.py
|   |-- linear_algebra.py
|   |-- preprocessing.py
|   |-- recommendation.py
|   |-- regression.py
|   |-- statistics.py
|-- app.py
|-- main.py
|-- README.md
|-- requirements.txt
```

## How To Run In VS Code

1. Open the `project` folder in VS Code.
2. Open a terminal in VS Code.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

5. Install requirements:

```bash
pip install -r requirements.txt
```

6. Run the project:

```bash
python main.py
```

This opens the professional browser UI with all project modules available from the sidebar.

Optional: you can also launch the same browser UI directly:

```bash
python web_app.py
```

## UI Sections

- Dashboard with a complete project summary button
- Linear Algebra
- Probability and Statistics
- Titanic Analysis
- Machine Learning Models
- Custom CSV Analysis

## How To Analyze Your Own CSV File

Open `Custom CSV` from the sidebar and click `Browse CSV`.

Then enter a CSV path. Examples:

```text
data/titanic.csv
```

or:

```text
C:\Users\YourName\Desktop\my_data.csv
```

After loading the CSV, you can perform:

- Dataset overview
- Numeric statistics
- Missing value handling
- Row filtering from a popup form
- Groupby aggregation from a popup form
- Histogram plotting
- Correlation analysis

## Dataset Loading Instructions

Sample CSV datasets are included in the `data` folder, so the project runs without any downloads. You can also load your own CSV file using option `6`.

## Sample Outputs

The UI lets you run:

- Linear algebra operations: determinant, inverse, and equation solving
- Statistics: mean, median, variance, standard deviation, histogram, and coin toss probabilities
- Titanic analysis: missing value handling, EDA, survival by gender, and survival by class
- Custom CSV analysis: overview, filtering, grouping, missing value handling, plots, and correlations
- Machine learning models:
  - House price prediction using Linear Regression
  - Spam detection using Logistic Regression
  - Movie recommendation and high/low rating classification
  - Uber fare prediction using Linear Regression and Random Forest

Charts and tables are displayed directly in the browser UI.

## Professional UI

The project includes a browser UI that runs locally from Python. It has pages for:

- Dashboard
- Linear Algebra
- Statistics
- Titanic Analysis
- Machine Learning Models
- Custom CSV Analysis

Run it with:

```bash
python main.py
```
