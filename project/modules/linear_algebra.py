"""Linear algebra examples using NumPy."""

from __future__ import annotations

import numpy as np


def create_matrix() -> np.ndarray:
    """Create a 3x3 matrix."""
    return np.array(
        [
            [2, 1, 3],
            [1, 0, 2],
            [4, 1, 8],
        ],
        dtype=float,
    )


def calculate_determinant(matrix: np.ndarray) -> float:
    """Calculate the determinant of a matrix."""
    return float(np.linalg.det(matrix))


def calculate_inverse(matrix: np.ndarray) -> np.ndarray | None:
    """Calculate the inverse of a matrix if it exists."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("Matrix inverse does not exist because the matrix is singular.")
        return None


def solve_equations() -> np.ndarray:
    """Solve 2x + 3y = 10 and 4x + 5y = 20."""
    coefficients = np.array([[2, 3], [4, 5]], dtype=float)
    constants = np.array([10, 20], dtype=float)
    return np.linalg.solve(coefficients, constants)


def run_linear_algebra_demo() -> None:
    """Run all linear algebra examples."""
    matrix = create_matrix()
    determinant = calculate_determinant(matrix)
    inverse = calculate_inverse(matrix)
    solution = solve_equations()

    print("\n--- Linear Algebra Module ---")
    print("3x3 Matrix:")
    print(matrix)
    print(f"Determinant: {determinant:.2f}")

    if inverse is not None:
        print("Inverse Matrix:")
        print(np.round(inverse, 2))

    print("Solution for equations 2x + 3y = 10 and 4x + 5y = 20:")
    print(f"x = {solution[0]:.2f}, y = {solution[1]:.2f}")
