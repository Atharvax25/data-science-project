"""Probability and statistics examples using NumPy and Matplotlib."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def generate_normal_distribution(size: int = 1000, mean: float = 50, std_dev: float = 10) -> np.ndarray:
    """Generate values from a normal distribution."""
    return np.random.default_rng(42).normal(loc=mean, scale=std_dev, size=size)


def calculate_statistics(data: np.ndarray) -> dict[str, float]:
    """Calculate common statistics."""
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "variance": float(np.var(data)),
        "standard_deviation": float(np.std(data)),
    }


def plot_histogram(data: np.ndarray) -> None:
    """Plot histogram for normally distributed data."""
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color="skyblue", edgecolor="black")
    plt.title("Normal Distribution Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def simulate_coin_tosses(tosses: int = 1000) -> dict[str, float | int]:
    """Simulate coin tosses and calculate probabilities."""
    rng = np.random.default_rng(42)
    results = rng.choice(["Heads", "Tails"], size=tosses)
    heads_count = int(np.sum(results == "Heads"))
    tails_count = int(np.sum(results == "Tails"))

    return {
        "total_tosses": tosses,
        "heads_count": heads_count,
        "tails_count": tails_count,
        "heads_probability": heads_count / tosses,
        "tails_probability": tails_count / tosses,
    }


def run_statistics_demo() -> None:
    """Run statistics and probability examples."""
    data = generate_normal_distribution()
    stats = calculate_statistics(data)
    coin_results = simulate_coin_tosses()

    print("\n--- Probability and Statistics Module ---")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Variance: {stats['variance']:.2f}")
    print(f"Standard Deviation: {stats['standard_deviation']:.2f}")

    print("\nCoin Toss Simulation:")
    print(f"Total Tosses: {coin_results['total_tosses']}")
    print(f"Heads: {coin_results['heads_count']}")
    print(f"Tails: {coin_results['tails_count']}")
    print(f"Heads Probability: {coin_results['heads_probability']:.2f}")
    print(f"Tails Probability: {coin_results['tails_probability']:.2f}")

    plot_histogram(data)
