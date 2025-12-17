"""
Cross-validation evaluation for DSL programs.

Evaluates compiled estimators using cross-validation and returns scores.
"""

from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from compiler import dsl_to_sklearn_estimator


def evaluate_program(
    dsl_string: str,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "accuracy",
    folds: int = 5,
    seed: int = 0,
    penalty_value: float = float("-inf"),
) -> float:
    """
    Evaluate a DSL program using cross-validation.
    
    Args:
        dsl_string: DSL string to evaluate
        X: Feature matrix
        y: Target vector
        scoring: Scoring metric (default: "accuracy")
        folds: Number of CV folds
        seed: Random seed for reproducibility
        penalty_value: Value to return on error (default: -inf)
        
    Returns:
        Mean CV score, or penalty_value on error
    """
    try:
        # Compile DSL to sklearn estimator
        est = dsl_to_sklearn_estimator(dsl_string)
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        
        # Evaluate
        scores = cross_val_score(
            est,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Use n_jobs=1 to avoid issues in some environments
        )
        
        return float(scores.mean())
    
    except Exception:
        # Return penalty value on any error
        return penalty_value

