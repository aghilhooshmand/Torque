"""
Registry for allowed models and ensembles.

This prevents arbitrary code execution and keeps the DSL safe.
Now includes a richer set of classical ML models.
Valid parameters are derived from sklearn API.
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


def _get_valid_params(cls, **init_kwargs):
    """Return set of valid init parameter names for a sklearn class."""
    try:
        inst = cls(**init_kwargs)
        return set(inst.get_params(deep=True).keys())
    except Exception:
        return set()


# Build VALID_PARAMS from sklearn (top-level init params only, no nested __)
def _build_valid_params():
    out = {}
    # Models
    for name, cls in [
        ("LR", LogisticRegression),
        ("DT", DecisionTreeClassifier),
        ("NB", GaussianNB),
        ("SVM", SVC),
        ("KNN", KNeighborsClassifier),
        ("RF", RandomForestClassifier),
        ("GB", GradientBoostingClassifier),
    ]:
        out[name] = _get_valid_params(cls)
    if XGBClassifier is not None:
        out["XGB"] = _get_valid_params(XGBClassifier)
    # Ensembles (need minimal init args)
    out["vote"] = _get_valid_params(VotingClassifier, estimators=[("d", LogisticRegression())])
    out["stack"] = _get_valid_params(
        StackingClassifier,
        estimators=[("d", LogisticRegression())],
        final_estimator=LogisticRegression(),
    )
    out["bag"] = _get_valid_params(BaggingClassifier, estimator=DecisionTreeClassifier())
    out["ada"] = _get_valid_params(AdaBoostClassifier, estimator=DecisionTreeClassifier())
    return out


VALID_PARAMS = _build_valid_params()

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

