"""
Registry for allowed models and ensembles.

This prevents arbitrary code execution and keeps the DSL safe.
"""

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Model registry - maps DSL model names to sklearn classes
MODEL_REGISTRY = {
    "LR": LogisticRegression,
    "SVM": SVC,
    "RF": RandomForestClassifier,
    "DT": DecisionTreeClassifier,
    "NB": GaussianNB,
    # Add more models as needed
}

# Ensemble registry - maps DSL ensemble names to sklearn classes
ENSEMBLE_REGISTRY = {
    "vote": VotingClassifier,
    "stack": StackingClassifier,
    "bag": BaggingClassifier,
    "ada": AdaBoostClassifier,
    # Add more ensembles as needed
}

# Default parameters for models (optional - can be used for convenience)
MODEL_DEFAULTS = {
    "LR": {"max_iter": 1000, "random_state": 42},
    "SVM": {"probability": True, "random_state": 42},
    "RF": {"random_state": 42},
    "DT": {"random_state": 42},
    "NB": {},
}

