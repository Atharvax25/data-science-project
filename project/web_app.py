"""Browser UI for the Smart Data Analytics and Prediction System.

The app uses only Python's standard library for serving the interface, so it
works even when Tkinter/Tcl is not installed correctly on the machine.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modules.custom_csv_analysis import clean_missing_values
from modules.linear_algebra import calculate_determinant, calculate_inverse, create_matrix, solve_equations
from modules.preprocessing import handle_titanic_missing_values, load_dataset
from modules.regression import preprocess_uber_data
from modules.statistics import calculate_statistics, generate_normal_distribution, simulate_coin_tosses


HOST = "127.0.0.1"
PORT = 8000


def json_ready(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_string()
    if isinstance(value, pd.Series):
        return value.to_string()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def frame_preview(df: pd.DataFrame, rows: int = 30) -> dict[str, Any]:
    """Return a compact table preview for the web UI."""
    preview = df.head(rows).replace({np.nan: None})
    return {
        "columns": list(preview.columns),
        "rows": preview.astype(object).where(pd.notnull(preview), None).values.tolist(),
    }


def text_response(title: str, body: str, **extra: Any) -> dict[str, Any]:
    payload = {"title": title, "body": body}
    payload.update(extra)
    return json_ready(payload)


def linear_algebra_payload() -> dict[str, Any]:
    matrix = create_matrix()
    inverse = calculate_inverse(matrix)
    solution = solve_equations()
    body = "\n".join(
        [
            "3x3 Matrix:",
            str(matrix),
            f"\nDeterminant: {calculate_determinant(matrix):.2f}",
            "\nInverse Matrix:",
            str(inverse.round(2)) if inverse is not None else "Inverse does not exist.",
            "\nEquation solution for 2x + 3y = 10 and 4x + 5y = 20:",
            f"x = {solution[0]:.2f}, y = {solution[1]:.2f}",
        ]
    )
    return text_response("Linear Algebra", body)


def statistics_payload() -> dict[str, Any]:
    data = generate_normal_distribution()
    stats = calculate_statistics(data)
    coin = simulate_coin_tosses()
    counts, edges = np.histogram(data, bins=24)
    body = (
        f"Mean: {stats['mean']:.2f}\n"
        f"Median: {stats['median']:.2f}\n"
        f"Variance: {stats['variance']:.2f}\n"
        f"Standard Deviation: {stats['standard_deviation']:.2f}\n\n"
        "Coin Toss Simulation:\n"
        f"Total Tosses: {coin['total_tosses']}\n"
        f"Heads: {coin['heads_count']}\n"
        f"Tails: {coin['tails_count']}\n"
        f"Heads Probability: {coin['heads_probability']:.2f}\n"
        f"Tails Probability: {coin['tails_probability']:.2f}"
    )
    return text_response(
        "Probability and Statistics",
        body,
        chart={"type": "bar", "labels": [f"{edges[i]:.1f}" for i in range(len(counts))], "values": counts.tolist()},
    )


def titanic_payload(chart: str = "gender") -> dict[str, Any]:
    df = handle_titanic_missing_values(load_dataset("titanic.csv"))
    numeric_df = df.select_dtypes(include="number")
    body = (
        f"Dataset shape: {df.shape}\n\n"
        f"Missing values after cleaning:\n{df.isnull().sum()}\n\n"
        f"Summary statistics:\n{numeric_df.describe()}\n\n"
        f"Average survival by gender:\n{df.groupby('Sex')['Survived'].mean()}\n\n"
        f"Fare aggregations by passenger class:\n{df.groupby('Pclass')['Fare'].agg(['sum', 'mean', 'min', 'max'])}"
    )
    if chart == "class":
        grouped = df.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
        labels = [f"Class {label}" for label in grouped.index.tolist()]
    else:
        grouped = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
        labels = grouped.index.tolist()
    no_values = grouped.get(0, pd.Series([0] * len(grouped), index=grouped.index)).tolist()
    yes_values = grouped.get(1, pd.Series([0] * len(grouped), index=grouped.index)).tolist()
    return text_response(
        "Titanic Analysis",
        body,
        chart={"type": "grouped", "labels": labels, "series": [{"name": "Not Survived", "values": no_values}, {"name": "Survived", "values": yes_values}]},
    )


def house_price_report() -> str:
    df = load_dataset("house_prices.csv")
    x = df[["area", "bedrooms", "age"]]
    y = df["price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    sample = pd.DataFrame([[2200, 3, 5]], columns=["area", "bedrooms", "age"])
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return (
        "--- House Price Prediction ---\n"
        f"R2 Score: {r2_score(y_test, predictions):.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"Predicted price for area=2200, bedrooms=3, age=5: {model.predict(sample)[0]:.2f}\n"
    )


def spam_report() -> str:
    df = load_dataset("emails.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = Pipeline([("vectorizer", CountVectorizer()), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    sample_prediction = model.predict(["Congratulations claim your free prize now"])[0]
    return (
        "--- Spam Detection ---\n"
        f"Accuracy: {accuracy_score(y_test, predictions):.2f}\n"
        f"{classification_report(y_test, predictions, zero_division=0)}\n"
        f"Sample email prediction: {sample_prediction}\n"
    )


def recommendation_report() -> str:
    df = load_dataset("movie_ratings.csv")
    scores = (
        df.groupby("movie_title")
        .agg(average_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
        .sort_values(by=["average_rating", "rating_count"], ascending=False)
    )
    df["rating_class"] = df["rating"].apply(lambda value: "high" if value >= 4 else "low")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x = encoder.fit_transform(df[["movie_title", "genre"]])
    y = df["rating_class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    predictions = model.predict(x_test)
    return (
        "--- Movie Recommendation ---\n"
        f"Recommended movies:\n{scores[scores['average_rating'] >= 4.0]}\n\n"
        "Rating Classification:\n"
        f"Accuracy: {accuracy_score(y_test, predictions):.2f}\n"
        f"{classification_report(y_test, predictions, zero_division=0)}"
    )


def uber_report() -> str:
    df = load_dataset("uber_fares.csv")
    cleaned_df = preprocess_uber_data(df)
    features = ["distance_km", "passenger_count", "hour", "day_of_week"]
    x = cleaned_df[features]
    y = cleaned_df["fare_amount"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    linear_model = LinearRegression().fit(x_train, y_train)
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(x_train, y_train)
    linear_predictions = linear_model.predict(x_test)
    forest_predictions = forest_model.predict(x_test)
    sample = pd.DataFrame([[6.0, 2, 18, 5]], columns=features)
    return (
        "--- Uber Fare Prediction ---\n"
        f"Rows before preprocessing: {len(df)}\n"
        f"Rows after preprocessing: {len(cleaned_df)}\n\n"
        "Linear Regression:\n"
        f"R2 Score: {r2_score(y_test, linear_predictions):.2f}\n"
        f"RMSE: {(mean_squared_error(y_test, linear_predictions) ** 0.5):.2f}\n\n"
        "Random Forest Regressor:\n"
        f"R2 Score: {r2_score(y_test, forest_predictions):.2f}\n"
        f"RMSE: {(mean_squared_error(y_test, forest_predictions) ** 0.5):.2f}\n"
        f"Sample ride predicted fare: {forest_model.predict(sample)[0]:.2f}\n"
    )


def ml_payload(model: str = "all") -> dict[str, Any]:
    reports = {
        "house": house_price_report,
        "spam": spam_report,
        "recommendation": recommendation_report,
        "uber": uber_report,
    }
    if model in reports:
        body = reports[model]()
    else:
        body = "\n".join(report() for report in reports.values())
    return text_response("Machine Learning Models", body)


def dashboard_payload() -> dict[str, Any]:
    body = "\n\n".join(
        [
            linear_algebra_payload()["body"],
            statistics_payload()["body"],
            titanic_payload()["body"],
            ml_payload()["body"],
        ]
    )
    return text_response("Complete Project Summary", body)


def read_csv_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    csv_text = str(payload.get("csv_text", "")).strip()
    if not csv_text:
        raise ValueError("Please select or paste a CSV file first.")
    return pd.read_csv(StringIO(csv_text))


def csv_payload(payload: dict[str, Any]) -> dict[str, Any]:
    df = read_csv_from_payload(payload)
    action = payload.get("action", "overview")

    if action == "stats":
        numeric_df = df.select_dtypes(include="number")
        body = "No numeric columns found." if numeric_df.empty else (
            f"--- Numeric Statistics ---\n{numeric_df.describe()}\n\n"
            f"Mean:\n{numeric_df.mean()}\n\nMedian:\n{numeric_df.median()}\n\n"
            f"Variance:\n{numeric_df.var()}\n\nStandard deviation:\n{numeric_df.std()}"
        )
        return text_response("CSV Statistics", body, table=frame_preview(df))

    if action == "clean":
        cleaned = clean_missing_values(df)
        return text_response("Clean Missing Values", str(cleaned.isnull().sum()), table=frame_preview(cleaned))

    if action == "filter":
        filtered = filter_dataframe(df, str(payload.get("column", "")), str(payload.get("operator", "=")), str(payload.get("value", "")))
        return text_response("Filtered Rows", f"Total rows: {len(filtered)}", table=frame_preview(filtered, 50))

    if action == "group":
        group_column = str(payload.get("group_column", ""))
        value_column = str(payload.get("value_column", ""))
        if group_column not in df.columns or value_column not in df.select_dtypes(include="number").columns:
            raise ValueError("Choose a valid group column and numeric value column.")
        result = df.groupby(group_column)[value_column].agg(["sum", "mean", "min", "max", "count"]).reset_index()
        return text_response("Group By Aggregation", result.to_string(index=False), table=frame_preview(result, 50))

    if action == "histogram":
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise ValueError("No numeric columns available for histogram.")
        column = str(payload.get("column") or numeric_columns[0])
        counts, edges = np.histogram(df[column].dropna(), bins=18)
        return text_response(
            "Histogram",
            f"Histogram created for {column}.",
            chart={"type": "bar", "labels": [f"{edges[i]:.1f}" for i in range(len(counts))], "values": counts.tolist()},
            table=frame_preview(df),
        )

    if action == "correlation":
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            raise ValueError("At least two numeric columns are needed for correlation analysis.")
        corr = numeric_df.corr().reset_index().rename(columns={"index": "column"})
        return text_response("Correlation Matrix", numeric_df.corr().to_string(), table=frame_preview(corr, 50))

    body = (
        f"Rows and columns: {df.shape}\n\n"
        f"Columns:\n{list(df.columns)}\n\n"
        f"Data types:\n{df.dtypes}\n\n"
        f"Missing values:\n{df.isnull().sum()}"
    )
    metadata = {
        "columns": list(df.columns),
        "numericColumns": df.select_dtypes(include="number").columns.tolist(),
    }
    return text_response("Dataset Overview", body, table=frame_preview(df), metadata=metadata)


def filter_dataframe(df: pd.DataFrame, column: str, operator: str, raw_value: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError("Column not found.")
    if raw_value == "":
        raise ValueError("Please enter a value.")
    value: object = raw_value
    if pd.api.types.is_numeric_dtype(df[column]):
        value = float(raw_value)
    if operator == "=":
        return df[df[column] == value]
    if operator == "!=":
        return df[df[column] != value]
    if operator == ">":
        return df[df[column] > value]
    if operator == "<":
        return df[df[column] < value]
    if operator == ">=":
        return df[df[column] >= value]
    if operator == "<=":
        return df[df[column] <= value]
    raise ValueError("Invalid operator.")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Data Analytics</title>
  <style>
    :root { --nav:#111827; --blue:#2563eb; --bg:#eef2f7; --panel:#fff; --text:#111827; --muted:#6b7280; --border:#d1d5db; --green:#059669; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--text); }
    .app { min-height:100vh; display:grid; grid-template-columns:260px 1fr; }
    aside { background:var(--nav); color:white; padding:22px 12px; }
    .brand { font-size:24px; font-weight:800; padding:8px 10px 26px; }
    .nav button { width:100%; border:0; background:transparent; color:#d1d5db; text-align:left; padding:13px 14px; border-radius:8px; font-size:15px; cursor:pointer; }
    .nav button:hover, .nav button.active { background:var(--blue); color:#fff; }
    main { padding:28px; min-width:0; }
    .topbar { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:18px; }
    h1 { margin:0; font-size:30px; line-height:1.1; }
    .subtitle { color:var(--muted); margin-top:6px; }
    .grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin-bottom:18px; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:18px; box-shadow:0 10px 30px rgba(15,23,42,.06); }
    .panel h2 { font-size:17px; margin:0 0 10px; }
    .muted { color:var(--muted); }
    .workspace { display:grid; grid-template-columns:minmax(0, 1.1fr) minmax(320px, .9fr); gap:16px; }
    pre { margin:0; white-space:pre-wrap; overflow:auto; max-height:560px; font:13px/1.5 Consolas, monospace; }
    button.primary, .actions button { border:0; border-radius:8px; background:var(--blue); color:#fff; padding:10px 13px; font-weight:600; cursor:pointer; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
    .actions button.secondary { background:#e5e7eb; color:#111827; }
    select, input[type="text"], input[type="file"] { width:100%; border:1px solid var(--border); border-radius:8px; padding:9px 10px; background:#fff; }
    label { display:block; font-size:13px; font-weight:700; margin:10px 0 5px; }
    canvas { width:100%; height:330px; border:1px solid var(--border); border-radius:8px; background:#fff; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; vertical-align:top; }
    th { background:#f8fafc; position:sticky; top:0; }
    .table-wrap { max-height:360px; overflow:auto; border:1px solid var(--border); border-radius:8px; }
    .notice { padding:10px 12px; border-radius:8px; background:#ecfdf5; color:#065f46; margin-bottom:12px; }
    @media (max-width:900px) { .app { grid-template-columns:1fr; } aside { position:static; } .grid, .workspace { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">Smart Analytics</div>
      <div class="nav" id="nav"></div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1 id="title">Dashboard</h1>
          <div class="subtitle" id="subtitle">Run analytics, visualize datasets, and train beginner-friendly machine learning models.</div>
        </div>
      </div>
      <section id="content"></section>
    </main>
  </div>
  <script>
    const pages = [
      ["dashboard", "Dashboard"], ["linear", "Linear Algebra"], ["statistics", "Statistics"],
      ["titanic", "Titanic Analysis"], ["ml", "ML Models"], ["csv", "Custom CSV"]
    ];
    let currentCsv = "";
    let csvColumns = [];
    let csvNumericColumns = [];

    const nav = document.getElementById("nav");
    pages.forEach(([id, label]) => {
      const button = document.createElement("button");
      button.textContent = label;
      button.onclick = () => showPage(id);
      button.id = "nav-" + id;
      nav.appendChild(button);
    });

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function setActive(id) {
      pages.forEach(([page]) => document.getElementById("nav-" + page).classList.toggle("active", page === id));
    }

    function shell(title, subtitle, inner) {
      document.getElementById("title").textContent = title;
      document.getElementById("subtitle").textContent = subtitle;
      document.getElementById("content").innerHTML = inner;
    }

    function resultLayout() {
      return `<div class="workspace"><div class="panel"><h2>Results</h2><div class="actions" id="actions"></div><pre id="output">Loading...</pre></div><div class="panel"><h2>Visualization</h2><canvas id="chart"></canvas><div id="table"></div></div></div>`;
    }

    function renderText(data) {
      document.getElementById("output").textContent = data.body || "";
      if (data.chart) drawChart(data.chart);
      if (data.table) renderTable(data.table);
    }

    function drawChart(chart) {
      const canvas = document.getElementById("chart");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const width = canvas.width = canvas.clientWidth * devicePixelRatio;
      const height = canvas.height = 330 * devicePixelRatio;
      ctx.scale(devicePixelRatio, devicePixelRatio);
      ctx.clearRect(0, 0, width, height);
      const pad = 42, plotW = canvas.clientWidth - pad - 16, plotH = 250;
      const values = chart.type === "grouped" ? chart.series.flatMap(s => s.values) : chart.values;
      const max = Math.max(...values, 1);
      ctx.strokeStyle = "#d1d5db"; ctx.beginPath(); ctx.moveTo(pad, 18); ctx.lineTo(pad, plotH + 18); ctx.lineTo(canvas.clientWidth - 16, plotH + 18); ctx.stroke();
      if (chart.type === "grouped") {
        const colors = ["#94a3b8", "#2563eb"];
        chart.labels.forEach((label, i) => {
          const groupW = plotW / chart.labels.length;
          chart.series.forEach((series, j) => {
            const barW = Math.max(12, groupW / 4);
            const h = (series.values[i] / max) * (plotH - 20);
            const x = pad + i * groupW + 12 + j * (barW + 6);
            ctx.fillStyle = colors[j]; ctx.fillRect(x, plotH + 18 - h, barW, h);
          });
          ctx.fillStyle = "#475569"; ctx.font = "12px Segoe UI"; ctx.fillText(label, pad + i * groupW + 8, plotH + 38);
        });
      } else {
        const barW = plotW / chart.values.length;
        chart.values.forEach((value, i) => {
          const h = (value / max) * (plotH - 20);
          ctx.fillStyle = "#2563eb"; ctx.fillRect(pad + i * barW + 2, plotH + 18 - h, Math.max(2, barW - 4), h);
        });
      }
    }

    function renderTable(table) {
      const target = document.getElementById("table");
      if (!target) return;
      const head = table.columns.map(c => `<th>${escapeHtml(c)}</th>`).join("");
      const rows = table.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell ?? "")}</td>`).join("")}</tr>`).join("");
      target.innerHTML = `<h2 style="margin-top:18px">Preview</h2><div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }

    async function showPage(id) {
      setActive(id);
      if (id === "dashboard") return showDashboard();
      if (id === "csv") return showCsv();
      shell(pages.find(p => p[0] === id)[1], "All project output stays inside this UI.", resultLayout());
      const actions = document.getElementById("actions");
      if (id === "titanic") {
        actions.innerHTML = `<button onclick="loadTitanic('gender')">Survival by Gender</button><button onclick="loadTitanic('class')" class="secondary">Survival by Class</button>`;
        return loadTitanic("gender");
      }
      if (id === "ml") {
        actions.innerHTML = ["all","house","spam","recommendation","uber"].map(m => `<button onclick="loadMl('${m}')" class="${m === 'all' ? '' : 'secondary'}">${m}</button>`).join("");
        return loadMl("all");
      }
      const data = await api(`/api/${id}`);
      renderText(data);
    }

    async function showDashboard() {
      shell("Dashboard", "Run the complete project summary or jump into a focused module.", `
        <div class="grid">
          <div class="panel"><h2>CSV Ready</h2><p class="muted">Upload your own data and run overview, stats, cleaning, filters, group by, histogram, and correlation.</p></div>
          <div class="panel"><h2>ML Examples</h2><p class="muted">House price prediction, spam detection, recommendations, and Uber fare prediction.</p></div>
          <div class="panel"><h2>Visual Output</h2><p class="muted">Charts render directly in the app, with readable reports beside them.</p></div>
        </div>${resultLayout()}`);
      document.getElementById("actions").innerHTML = `<button onclick="loadDashboard()">Run Complete Project</button>`;
      document.getElementById("output").textContent = "Click Run Complete Project to generate the full report.";
      document.getElementById("chart").style.display = "none";
    }

    async function loadDashboard() { renderText(await api("/api/dashboard")); }
    async function loadTitanic(chart) { renderText(await api(`/api/titanic?chart=${chart}`)); }
    async function loadMl(model) { renderText(await api(`/api/ml?model=${model}`)); }

    function showCsv() {
      shell("Custom CSV Analysis", "Upload your dataset and analyze it without leaving the UI.", `
        <div class="workspace">
          <div class="panel">
            <h2>Dataset Controls</h2>
            <input id="csvFile" type="file" accept=".csv">
            <div class="actions" style="margin-top:12px">
              <button onclick="runCsv('overview')">Overview</button>
              <button onclick="runCsv('stats')" class="secondary">Statistics</button>
              <button onclick="runCsv('clean')" class="secondary">Clean Missing</button>
              <button onclick="runCsv('histogram')" class="secondary">Histogram</button>
              <button onclick="runCsv('correlation')" class="secondary">Correlation</button>
            </div>
            <label>Filter rows</label>
            <div class="grid" style="grid-template-columns:1fr 110px 1fr;margin-bottom:10px">
              <select id="filterColumn"></select><select id="filterOperator"><option>=</option><option>!=</option><option>&gt;</option><option>&lt;</option><option>&gt;=</option><option>&lt;=</option></select><input id="filterValue" type="text" placeholder="Value">
            </div>
            <button class="primary" onclick="runCsv('filter')">Apply Filter</button>
            <label>Group by aggregation</label>
            <div class="grid" style="grid-template-columns:1fr 1fr;margin-bottom:10px">
              <select id="groupColumn"></select><select id="valueColumn"></select>
            </div>
            <button class="primary" onclick="runCsv('group')">Run Group By</button>
            <pre id="output" style="margin-top:14px">Choose a CSV file to begin.</pre>
          </div>
          <div class="panel"><h2>Visualization and Preview</h2><canvas id="chart"></canvas><div id="table"></div></div>
        </div>`);
      document.getElementById("csvFile").addEventListener("change", event => {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => { currentCsv = reader.result; runCsv("overview"); };
        reader.readAsText(file);
      });
    }

    async function runCsv(action) {
      if (!currentCsv) { document.getElementById("output").textContent = "Please choose a CSV file first."; return; }
      const payload = { csv_text: currentCsv, action };
      if (action === "filter") {
        payload.column = document.getElementById("filterColumn").value;
        payload.operator = document.getElementById("filterOperator").value;
        payload.value = document.getElementById("filterValue").value;
      }
      if (action === "group") {
        payload.group_column = document.getElementById("groupColumn").value;
        payload.value_column = document.getElementById("valueColumn").value;
      }
      try {
        const data = await api("/api/csv", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
        renderText(data);
        if (data.metadata) {
          csvColumns = data.metadata.columns; csvNumericColumns = data.metadata.numericColumns;
          fillSelect("filterColumn", csvColumns); fillSelect("groupColumn", csvColumns); fillSelect("valueColumn", csvNumericColumns);
        }
      } catch (error) { document.getElementById("output").textContent = error.message; }
    }

    function fillSelect(id, values) {
      const select = document.getElementById(id);
      if (!select) return;
      select.innerHTML = values.map(value => `<option>${escapeHtml(value)}</option>`).join("");
    }

    showPage("dashboard");
  </script>
</body>
</html>
"""


class AnalyticsRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local browser UI."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(json_ready(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_html(INDEX_HTML)
            elif parsed.path == "/api/dashboard":
                self.send_json(dashboard_payload())
            elif parsed.path == "/api/linear":
                self.send_json(linear_algebra_payload())
            elif parsed.path == "/api/statistics":
                self.send_json(statistics_payload())
            elif parsed.path == "/api/titanic":
                self.send_json(titanic_payload(query.get("chart", ["gender"])[0]))
            elif parsed.path == "/api/ml":
                self.send_json(ml_payload(query.get("model", ["all"])[0]))
            else:
                self.send_json({"error": "Not found"}, 404)
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if self.path == "/api/csv":
                self.send_json(csv_payload(payload))
            else:
                self.send_json({"error": "Not found"}, 404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)


def main() -> None:
    """Start the local web app."""
    port = find_available_port(PORT)
    server = ThreadingHTTPServer((HOST, port), AnalyticsRequestHandler)
    url = f"http://{HOST}:{port}"
    print(f"Smart Analytics UI is running at {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Smart Analytics UI.")
    finally:
        server.server_close()


def find_available_port(start_port: int) -> int:
    """Find an open local port, starting with the preferred project port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((HOST, port)) != 0:
                return port
    raise OSError("No available local port found for the UI.")


if __name__ == "__main__":
    main()
