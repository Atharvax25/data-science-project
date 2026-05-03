"""Professional Tkinter UI for the Smart Data Analytics and Prediction System."""

from __future__ import annotations

import tkinter as tk
from io import StringIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modules.custom_csv_analysis import clean_missing_values, load_custom_csv
from modules.linear_algebra import calculate_determinant, calculate_inverse, create_matrix, solve_equations
from modules.preprocessing import handle_titanic_missing_values, load_dataset
from modules.regression import preprocess_uber_data
from modules.statistics import calculate_statistics, generate_normal_distribution, simulate_coin_tosses


class AnalyticsApp(tk.Tk):
    """Main desktop application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Data Analytics and Prediction System")
        self.geometry("1180x760")
        self.minsize(1050, 680)
        self.configure(bg="#eef2f7")

        self.custom_df: pd.DataFrame | None = None
        self.current_chart: FigureCanvasTkAgg | None = None

        self.colors = {
            "nav": "#111827",
            "nav_active": "#2563eb",
            "bg": "#eef2f7",
            "panel": "#ffffff",
            "text": "#111827",
            "muted": "#6b7280",
            "border": "#d1d5db",
        }

        self._configure_styles()
        self._build_layout()
        self.show_dashboard()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 13, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Primary.TButton", background="#2563eb", foreground="white", borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure("TCombobox", padding=6)

    def _build_layout(self) -> None:
        self.sidebar = tk.Frame(self, bg=self.colors["nav"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Label(
            self.sidebar,
            text="Smart Analytics",
            bg=self.colors["nav"],
            fg="white",
            font=("Segoe UI", 18, "bold"),
            anchor="w",
            padx=22,
            pady=24,
        )
        brand.pack(fill="x")

        self.nav_buttons: dict[str, tk.Button] = {}
        nav_items = [
            ("Dashboard", self.show_dashboard),
            ("Linear Algebra", self.show_linear_algebra),
            ("Statistics", self.show_statistics),
            ("Titanic Analysis", self.show_titanic),
            ("ML Models", self.show_ml_models),
            ("Custom CSV", self.show_custom_csv),
        ]

        for label, command in nav_items:
            button = tk.Button(
                self.sidebar,
                text=label,
                command=lambda name=label, action=command: self._navigate(name, action),
                bg=self.colors["nav"],
                fg="#d1d5db",
                activebackground=self.colors["nav_active"],
                activeforeground="white",
                bd=0,
                font=("Segoe UI", 11),
                anchor="w",
                padx=24,
                pady=13,
                cursor="hand2",
            )
            button.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[label] = button

        self.main_area = ttk.Frame(self, style="TFrame")
        self.main_area.pack(side="left", fill="both", expand=True)

        self.header = ttk.Frame(self.main_area, style="TFrame")
        self.header.pack(fill="x", padx=28, pady=(24, 12))

        self.title_label = ttk.Label(self.header, text="", style="Title.TLabel")
        self.title_label.pack(anchor="w")
        self.subtitle_label = ttk.Label(self.header, text="", style="Subtitle.TLabel")
        self.subtitle_label.pack(anchor="w", pady=(4, 0))

        self.content = ttk.Frame(self.main_area, style="TFrame")
        self.content.pack(fill="both", expand=True, padx=28, pady=(0, 24))

    def _navigate(self, name: str, command) -> None:
        for label, button in self.nav_buttons.items():
            is_active = label == name
            button.configure(bg=self.colors["nav_active"] if is_active else self.colors["nav"], fg="white" if is_active else "#d1d5db")
        command()

    def _clear_content(self, title: str, subtitle: str) -> None:
        self.title_label.configure(text=title)
        self.subtitle_label.configure(text=subtitle)
        for widget in self.content.winfo_children():
            widget.destroy()
        self.current_chart = None

    def _make_panel(self, parent, title: str) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        header = ttk.Label(panel, text=title, style="PanelTitle.TLabel")
        header.pack(anchor="w", pady=(0, 10))
        return panel

    def _make_output(self, parent) -> ScrolledText:
        output = ScrolledText(parent, height=18, wrap="word", font=("Consolas", 10), bd=1, relief="solid")
        output.pack(fill="both", expand=True)
        return output

    def _set_output(self, output: ScrolledText, text: str) -> None:
        output.configure(state="normal")
        output.delete("1.0", tk.END)
        output.insert(tk.END, text)
        output.configure(state="disabled")

    def _plot_figure(self, parent, figure: Figure) -> None:
        if self.current_chart is not None:
            self.current_chart.get_tk_widget().destroy()
        self.current_chart = FigureCanvasTkAgg(figure, master=parent)
        self.current_chart.draw()
        self.current_chart.get_tk_widget().pack(fill="both", expand=True)

    def _capture_output(self, function) -> str:
        buffer = StringIO()
        with redirect_stdout(buffer):
            function()
        return buffer.getvalue()

    def show_dashboard(self) -> None:
        self._clear_content(
            "Dashboard",
            "Run analytics, visualize datasets, and train beginner-friendly machine learning models.",
        )

        cards = ttk.Frame(self.content, style="TFrame")
        cards.pack(fill="x")

        card_data = [
            ("CSV Ready", "Upload your own CSV and run statistics, filtering, grouping, and charts."),
            ("Data Science Stack", "Built with NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn."),
            ("ML Examples", "House price prediction, spam detection, recommendations, and Uber fare prediction."),
        ]

        for index, (title, description) in enumerate(card_data):
            card = self._make_panel(cards, title)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 12, 0))
            ttk.Label(card, text=description, background="white", foreground=self.colors["muted"], wraplength=230).pack(anchor="w")
            cards.columnconfigure(index, weight=1)

        quick = self._make_panel(self.content, "Quick Start")
        quick.pack(fill="both", expand=True, pady=(18, 0))
        actions = ttk.Frame(quick, style="Panel.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        output = self._make_output(quick)
        ttk.Button(
            actions,
            text="Run Complete Project",
            style="Primary.TButton",
            command=lambda: self._set_output(output, self._full_project_report()),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Open Custom CSV", command=lambda: self._navigate("Custom CSV", self.show_custom_csv)).pack(side="left")
        self._set_output(
            output,
            "Everything is available from this UI. Use the sidebar for focused modules, or run the complete project summary here.",
        )

    def show_linear_algebra(self) -> None:
        self._clear_content("Linear Algebra", "Matrix determinant, inverse, and system of equations.")
        panel = self._make_panel(self.content, "Results")
        panel.pack(fill="both", expand=True)
        output = self._make_output(panel)

        def run() -> None:
            matrix = create_matrix()
            inverse = calculate_inverse(matrix)
            solution = solve_equations()
            text = [
                "3x3 Matrix:",
                str(matrix),
                f"\nDeterminant: {calculate_determinant(matrix):.2f}",
                "\nInverse Matrix:",
                str(inverse.round(2)) if inverse is not None else "Inverse does not exist.",
                "\nEquation solution for 2x + 3y = 10 and 4x + 5y = 20:",
                f"x = {solution[0]:.2f}, y = {solution[1]:.2f}",
            ]
            self._set_output(output, "\n".join(text))

        ttk.Button(panel, text="Run Linear Algebra", style="Primary.TButton", command=run).pack(anchor="w", pady=(0, 10))
        run()

    def show_statistics(self) -> None:
        self._clear_content("Statistics", "Normal distribution, descriptive statistics, histogram, and coin toss simulation.")
        top = ttk.Frame(self.content, style="TFrame")
        top.pack(fill="both", expand=True)

        result_panel = self._make_panel(top, "Results")
        result_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        chart_panel = self._make_panel(top, "Histogram")
        chart_panel.pack(side="left", fill="both", expand=True)
        output = self._make_output(result_panel)

        def run() -> None:
            data = generate_normal_distribution()
            stats = calculate_statistics(data)
            coin = simulate_coin_tosses()
            text = (
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
            self._set_output(output, text)

            figure = Figure(figsize=(5, 4), dpi=100)
            axis = figure.add_subplot(111)
            axis.hist(data, bins=30, color="#60a5fa", edgecolor="#1f2937")
            axis.set_title("Normal Distribution Histogram")
            axis.set_xlabel("Value")
            axis.set_ylabel("Frequency")
            figure.tight_layout()
            self._plot_figure(chart_panel, figure)

        ttk.Button(result_panel, text="Run Statistics", style="Primary.TButton", command=run).pack(anchor="w", pady=(0, 10))
        run()

    def show_titanic(self) -> None:
        self._clear_content("Titanic Analysis", "Missing value handling, EDA, survival charts, and grouped insights.")
        top = ttk.Frame(self.content, style="TFrame")
        top.pack(fill="both", expand=True)

        result_panel = self._make_panel(top, "EDA Output")
        result_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        chart_panel = self._make_panel(top, "Visualization")
        chart_panel.pack(side="left", fill="both", expand=True)
        output = self._make_output(result_panel)

        selector = ttk.Frame(result_panel, style="Panel.TFrame")
        selector.pack(fill="x", pady=(0, 10))
        chart_choice = tk.StringVar(value="Survival vs Gender")
        combo = ttk.Combobox(selector, textvariable=chart_choice, values=["Survival vs Gender", "Survival vs Class"], state="readonly")
        combo.pack(side="left", padx=(0, 8))

        def run() -> None:
            df = handle_titanic_missing_values(load_dataset("titanic.csv"))
            numeric_df = df.select_dtypes(include="number")
            text = (
                f"Dataset shape: {df.shape}\n\n"
                f"Missing values after cleaning:\n{df.isnull().sum()}\n\n"
                f"Summary statistics:\n{numeric_df.describe()}\n\n"
                f"Average survival by gender:\n{df.groupby('Sex')['Survived'].mean()}\n\n"
                f"Fare aggregations by passenger class:\n{df.groupby('Pclass')['Fare'].agg(['sum', 'mean', 'min', 'max'])}"
            )
            self._set_output(output, text)
            self._plot_titanic_chart(chart_panel, df, chart_choice.get())

        ttk.Button(selector, text="Show Chart", style="Primary.TButton", command=run).pack(side="left")
        run()

    def _plot_titanic_chart(self, parent, df: pd.DataFrame, chart_name: str) -> None:
        figure = Figure(figsize=(5, 4), dpi=100)
        axis = figure.add_subplot(111)
        if chart_name == "Survival vs Gender":
            sns.countplot(data=df, x="Sex", hue="Survived", palette="Set2", ax=axis)
            axis.set_title("Survival vs Gender")
        else:
            sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1", ax=axis)
            axis.set_title("Survival vs Passenger Class")
        axis.set_ylabel("Passenger Count")
        figure.tight_layout()
        self._plot_figure(parent, figure)

    def show_ml_models(self) -> None:
        self._clear_content("Machine Learning Models", "Train and evaluate prediction and classification models.")
        panel = self._make_panel(self.content, "Model Results")
        panel.pack(fill="both", expand=True)
        output = self._make_output(panel)

        buttons = ttk.Frame(panel, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(0, 10))

        ttk.Button(buttons, text="House Price", command=lambda: self._set_output(output, self._house_price_report())).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Spam Detection", command=lambda: self._set_output(output, self._spam_report())).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Recommendation", command=lambda: self._set_output(output, self._recommendation_report())).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Uber Fare", command=lambda: self._set_output(output, self._uber_report())).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Run All", style="Primary.TButton", command=lambda: self._set_output(output, self._all_ml_reports())).pack(side="left")
        self._set_output(output, self._all_ml_reports())

    def _house_price_report(self) -> str:
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

    def _spam_report(self) -> str:
        df = load_dataset("emails.csv")
        x_train, x_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.25,
            random_state=42,
            stratify=df["label"],
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

    def _recommendation_report(self) -> str:
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

    def _uber_report(self) -> str:
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

    def _all_ml_reports(self) -> str:
        return "\n".join([self._house_price_report(), self._spam_report(), self._recommendation_report(), self._uber_report()])

    def _full_project_report(self) -> str:
        matrix = create_matrix()
        inverse = calculate_inverse(matrix)
        solution = solve_equations()
        data = generate_normal_distribution()
        stats = calculate_statistics(data)
        coin = simulate_coin_tosses()
        titanic = handle_titanic_missing_values(load_dataset("titanic.csv"))
        numeric_titanic = titanic.select_dtypes(include="number")

        sections = [
            "--- Linear Algebra ---",
            f"Matrix:\n{matrix}",
            f"Determinant: {calculate_determinant(matrix):.2f}",
            f"Inverse:\n{inverse.round(2) if inverse is not None else 'Inverse does not exist.'}",
            f"Equation solution: x = {solution[0]:.2f}, y = {solution[1]:.2f}",
            "\n--- Probability and Statistics ---",
            f"Mean: {stats['mean']:.2f}",
            f"Median: {stats['median']:.2f}",
            f"Variance: {stats['variance']:.2f}",
            f"Standard Deviation: {stats['standard_deviation']:.2f}",
            f"Coin Tosses: {coin['heads_count']} heads, {coin['tails_count']} tails",
            "\n--- Titanic Analysis ---",
            f"Dataset shape: {titanic.shape}",
            f"Missing values after cleaning:\n{titanic.isnull().sum()}",
            f"Summary statistics:\n{numeric_titanic.describe()}",
            f"Average survival by gender:\n{titanic.groupby('Sex')['Survived'].mean()}",
            "\n--- Machine Learning Models ---",
            self._all_ml_reports(),
        ]
        return "\n".join(str(section) for section in sections)

    def show_custom_csv(self) -> None:
        self._clear_content("Custom CSV Analysis", "Upload your own CSV and perform automatic data analysis.")
        top = ttk.Frame(self.content, style="TFrame")
        top.pack(fill="both", expand=True)

        result_panel = self._make_panel(top, "CSV Output")
        result_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        chart_panel = self._make_panel(top, "Chart")
        chart_panel.pack(side="left", fill="both", expand=True)
        output = self._make_output(result_panel)

        controls = ttk.Frame(result_panel, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 10))
        path_var = tk.StringVar(value="No CSV selected")
        ttk.Label(controls, textvariable=path_var, background="white", foreground=self.colors["muted"]).pack(side="left", fill="x", expand=True)

        def browse() -> None:
            file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return
            try:
                self.custom_df = load_custom_csv(file_path)
                path_var.set(file_path)
                self._set_output(output, self._custom_overview(self.custom_df))
            except Exception as error:
                messagebox.showerror("CSV Error", str(error))

        ttk.Button(controls, text="Browse CSV", style="Primary.TButton", command=browse).pack(side="right", padx=(8, 0))

        actions = ttk.Frame(result_panel, style="Panel.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="Overview", command=lambda: self._run_csv_action(output, self._custom_overview)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Statistics", command=lambda: self._run_csv_action(output, self._custom_stats)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Clean Missing", command=lambda: self._clean_csv(output)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Filter Rows", command=lambda: self._open_filter_dialog(output)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Group By", command=lambda: self._open_groupby_dialog(output)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Histogram", command=lambda: self._plot_custom_histogram(output, chart_panel)).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Correlation", command=lambda: self._plot_custom_correlation(output, chart_panel)).pack(side="left")

        self._set_output(output, "Click 'Browse CSV' to load your dataset.")

    def _require_custom_df(self) -> pd.DataFrame | None:
        if self.custom_df is None:
            messagebox.showinfo("No CSV Selected", "Please browse and select a CSV file first.")
            return None
        return self.custom_df

    def _run_csv_action(self, output: ScrolledText, action) -> None:
        df = self._require_custom_df()
        if df is not None:
            self._set_output(output, action(df))

    def _custom_overview(self, df: pd.DataFrame) -> str:
        return (
            "--- Dataset Overview ---\n"
            f"Rows and columns: {df.shape}\n\n"
            f"Columns:\n{list(df.columns)}\n\n"
            f"First five rows:\n{df.head()}\n\n"
            f"Data types:\n{df.dtypes}\n\n"
            f"Missing values:\n{df.isnull().sum()}"
        )

    def _custom_stats(self, df: pd.DataFrame) -> str:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return "No numeric columns found."
        return (
            "--- Numeric Statistics ---\n"
            f"{numeric_df.describe()}\n\n"
            f"Mean:\n{numeric_df.mean()}\n\n"
            f"Median:\n{numeric_df.median()}\n\n"
            f"Variance:\n{numeric_df.var()}\n\n"
            f"Standard deviation:\n{numeric_df.std()}"
        )

    def _clean_csv(self, output: ScrolledText) -> None:
        df = self._require_custom_df()
        if df is None:
            return
        self.custom_df = clean_missing_values(df)
        self._set_output(output, "Missing values handled successfully.\n\n" + str(self.custom_df.isnull().sum()))

    def _open_filter_dialog(self, output: ScrolledText) -> None:
        df = self._require_custom_df()
        if df is None:
            return

        dialog = tk.Toplevel(self)
        dialog.title("Filter Rows")
        dialog.geometry("420x230")
        dialog.resizable(False, False)
        dialog.configure(bg=self.colors["panel"])
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, style="Panel.TFrame", padding=18)
        frame.pack(fill="both", expand=True)

        column_var = tk.StringVar(value=df.columns[0])
        operator_var = tk.StringVar(value="=")
        value_var = tk.StringVar()

        ttk.Label(frame, text="Column", background="white").pack(anchor="w")
        ttk.Combobox(frame, textvariable=column_var, values=list(df.columns), state="readonly").pack(fill="x", pady=(3, 10))
        ttk.Label(frame, text="Operator", background="white").pack(anchor="w")
        ttk.Combobox(frame, textvariable=operator_var, values=["=", "!=", ">", "<", ">=", "<="], state="readonly").pack(fill="x", pady=(3, 10))
        ttk.Label(frame, text="Value", background="white").pack(anchor="w")
        ttk.Entry(frame, textvariable=value_var).pack(fill="x", pady=(3, 14))

        def apply_filter() -> None:
            try:
                result = self._filter_dataframe(df, column_var.get(), operator_var.get(), value_var.get())
                self._set_output(output, f"--- Filtered Rows ---\nTotal rows: {len(result)}\n\n{result.head(50)}")
                dialog.destroy()
            except Exception as error:
                messagebox.showerror("Filter Error", str(error), parent=dialog)

        ttk.Button(frame, text="Apply Filter", style="Primary.TButton", command=apply_filter).pack(anchor="e")

    def _filter_dataframe(self, df: pd.DataFrame, column: str, operator: str, raw_value: str) -> pd.DataFrame:
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

    def _open_groupby_dialog(self, output: ScrolledText) -> None:
        df = self._require_custom_df()
        if df is None:
            return

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            self._set_output(output, "No numeric columns available for group by aggregation.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Group By Aggregation")
        dialog.geometry("420x190")
        dialog.resizable(False, False)
        dialog.configure(bg=self.colors["panel"])
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, style="Panel.TFrame", padding=18)
        frame.pack(fill="both", expand=True)

        group_var = tk.StringVar(value=df.columns[0])
        value_var = tk.StringVar(value=numeric_columns[0])

        ttk.Label(frame, text="Group column", background="white").pack(anchor="w")
        ttk.Combobox(frame, textvariable=group_var, values=list(df.columns), state="readonly").pack(fill="x", pady=(3, 10))
        ttk.Label(frame, text="Numeric value column", background="white").pack(anchor="w")
        ttk.Combobox(frame, textvariable=value_var, values=numeric_columns, state="readonly").pack(fill="x", pady=(3, 14))

        def apply_groupby() -> None:
            result = df.groupby(group_var.get())[value_var.get()].agg(["sum", "mean", "min", "max", "count"])
            self._set_output(output, f"--- Group By Aggregation ---\n\n{result}")
            dialog.destroy()

        ttk.Button(frame, text="Run Group By", style="Primary.TButton", command=apply_groupby).pack(anchor="e")

    def _plot_custom_histogram(self, output: ScrolledText, chart_panel: ttk.Frame) -> None:
        df = self._require_custom_df()
        if df is None:
            return
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            self._set_output(output, "No numeric columns available for histogram.")
            return

        column = numeric_columns[0]
        figure = Figure(figsize=(5, 4), dpi=100)
        axis = figure.add_subplot(111)
        sns.histplot(df[column].dropna(), kde=True, ax=axis, color="#2563eb")
        axis.set_title(f"Histogram of {column}")
        figure.tight_layout()
        self._plot_figure(chart_panel, figure)
        self._set_output(output, f"Histogram created for numeric column: {column}")

    def _plot_custom_correlation(self, output: ScrolledText, chart_panel: ttk.Frame) -> None:
        df = self._require_custom_df()
        if df is None:
            return
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            self._set_output(output, "At least two numeric columns are needed for correlation analysis.")
            return

        figure = Figure(figsize=(5, 4), dpi=100)
        axis = figure.add_subplot(111)
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=axis)
        axis.set_title("Correlation Heatmap")
        figure.tight_layout()
        self._plot_figure(chart_panel, figure)
        self._set_output(output, "--- Correlation Matrix ---\n" + str(numeric_df.corr()))


def main() -> None:
    """Run the Tkinter application."""
    app = AnalyticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
