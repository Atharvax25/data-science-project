"""Simple movie recommendation and rating classification."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from modules.preprocessing import load_dataset


def recommend_movies(min_rating: float = 4.0) -> pd.DataFrame:
    """Recommend movies with average ratings greater than or equal to min_rating."""
    df = load_dataset("movie_ratings.csv")
    movie_scores = (
        df.groupby("movie_title")
        .agg(average_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
        .sort_values(by=["average_rating", "rating_count"], ascending=False)
    )
    return movie_scores[movie_scores["average_rating"] >= min_rating]


def classify_ratings() -> None:
    """Classify movie ratings as high or low using Logistic Regression."""
    df = load_dataset("movie_ratings.csv")
    df["rating_class"] = df["rating"].apply(lambda value: "high" if value >= 4 else "low")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = encoder.fit_transform(df[["movie_title", "genre"]])
    x = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["movie_title", "genre"]))
    y = df["rating_class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nRating Classification:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions, zero_division=0))


def run_recommendation_system() -> None:
    """Run movie recommendation and rating classification."""
    recommendations = recommend_movies()
    print("\n--- Recommendation System ---")
    print("Recommended movies:")
    print(recommendations)
    classify_ratings()
