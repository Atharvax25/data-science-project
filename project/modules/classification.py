"""Classification examples using Logistic Regression."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modules.preprocessing import load_dataset


def run_spam_detection() -> None:
    """Classify emails as spam or not spam."""
    df = load_dataset("emails.csv")
    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    sample_email = ["Congratulations claim your free prize now"]
    sample_prediction = model.predict(sample_email)[0]

    print("\n--- Logistic Regression: Spam Detection ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Sample email prediction: {sample_prediction}")
