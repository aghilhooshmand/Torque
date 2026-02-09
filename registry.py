"""
Registry for allowed models and ensembles.

This prevents arbitrary code execution and keeps the DSL safe.
Now includes a richer set of classical ML models.
"""

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - xgboost may be optional
    XGBClassifier = None  # type: ignore

# Model registry - classical models (LR, DT, NB, SVM, KNN, RF, GB, XGB)
MODEL_REGISTRY = {
    "LR": LogisticRegression,
    "DT": DecisionTreeClassifier,
    "NB": GaussianNB,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RF": RandomForestClassifier,
    "GB": GradientBoostingClassifier,
}

if XGBClassifier is not None:
    MODEL_REGISTRY["XGB"] = XGBClassifier

# Ensemble registry - maps DSL ensemble names to sklearn classes
ENSEMBLE_REGISTRY = {
    "vote": VotingClassifier,
    "stack": StackingClassifier,
    "bag": BaggingClassifier,
    "ada": AdaBoostClassifier,
}

MODEL_DEFAULTS = {
    "LR": {"max_iter": 1000, "random_state": 42},
    "DT": {"random_state": 42},
    "NB": {},
    "SVM": {},
    "KNN": {},
    "RF": {"n_estimators": 100, "random_state": 42},
    "GB": {"random_state": 42},
    "XGB": {"n_estimators": 100, "random_state": 42} if XGBClassifier is not None else {},
}

