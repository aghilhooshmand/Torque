"""
Execution module for running compiled DSL models.

Provides utilities to execute DSL programs and run sklearn models.
"""

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import modules - handle both package and standalone execution
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compiler import compile_ast_to_estimator, dsl_to_sklearn_estimator
from dsl_parser import parse_dsl_to_ast
from evaluator import evaluate_program


def execute_dsl(
    dsl_string: str,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    return_model: bool = False,
) -> Tuple[float, Optional[Any]]:
    """
    Parse, compile, and execute a DSL program on data (train/test split).
    
    Args:
        dsl_string: The DSL string to execute
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        return_model: Whether to return the fitted model
        
    Returns:
        Tuple of (accuracy, model) where model is None if return_model=False
    """
    # Compile DSL to sklearn estimator
    model = dsl_to_sklearn_estimator(dsl_string)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if return_model:
        return accuracy, model
    else:
        return accuracy, None


def compile_dsl(dsl_string: str) -> Any:
    """
    Parse and compile a DSL string into a sklearn estimator (without fitting).
    
    Args:
        dsl_string: The DSL string to compile
        
    Returns:
        A compiled sklearn estimator (not fitted)
    """
    return dsl_to_sklearn_estimator(dsl_string)

