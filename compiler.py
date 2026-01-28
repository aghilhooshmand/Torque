"""
Compiler that converts dict-based AST nodes into scikit-learn estimators.

This is the "interpreter" that transforms the AST into executable sklearn objects.
"""

from typing import Any, Dict

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Import modules - handle both package and standalone execution
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from registry import ENSEMBLE_REGISTRY, MODEL_REGISTRY


def eval_value(node: dict) -> Any:
    """
    Convert a literal node or nested call to a Python value.
    
    Args:
        node: AST node (dict with "type" key)
        
    Returns:
        Python primitive value or compiled estimator
    """
    if node["type"] == "literal":
        return node["value"]
    
    if node["type"] == "call":
        # Nested call used as value (optional feature)
        return compile_ast_to_estimator(node)
    
    raise TypeError(f"Unknown node type: {node.get('type')}")


def safe_kwargs_for(cls, kwargs: dict) -> dict:
    """
    Sanitize kwargs to only include valid parameters for the class.
    
    This prevents crashes from unknown parameters in generated programs.
    
    Args:
        cls: sklearn class
        kwargs: Dictionary of keyword arguments
        
    Returns:
        Filtered dictionary with only valid parameters
    """
    try:
        # Create a temporary instance to get valid parameters
        # Use minimal defaults to avoid errors
        temp_instance = cls()
        valid_params = set(temp_instance.get_params(deep=True).keys())
        
        # Filter kwargs to only include valid parameters
        safe_kw = {k: v for k, v in kwargs.items() if k in valid_params}
        return safe_kw
    except Exception:
        # If we can't create instance, return empty dict or original
        # This is a fallback for edge cases
        return {}


def build_model(name: str, kw_nodes: dict, context: dict = None) -> Any:
    """
    Build a sklearn model from AST node.
    
    Args:
        name: Model name (e.g., "LR", "SVM")
        kw_nodes: Dictionary of keyword argument nodes
        context: Context dictionary (e.g., {"voting": "soft"})
        
    Returns:
        Compiled sklearn estimator
    """
    if context is None:
        context = {}
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    
    cls = MODEL_REGISTRY[name]
    
    # Convert AST nodes to Python values
    kw = {k: eval_value(v) for k, v in kw_nodes.items()}
    
    # Special rule: if soft voting, ensure SVM has probability=True
    if name == "SVM" and context.get("voting") == "soft":
        kw["probability"] = True
    
    # Special handling for LR penalty/solver constraint
    # penalty='l1' requires solver='liblinear' or 'saga'
    # penalty='elasticnet' requires solver='saga' and l1_ratio parameter
    if name == "LR":
        penalty = kw.get("penalty", "l2")
        if penalty == "l1" and "solver" not in kw:
            kw["solver"] = "liblinear"  # Use liblinear as default for l1 penalty
        elif penalty == "elasticnet":
            if "solver" not in kw:
                kw["solver"] = "saga"  # Use saga for elasticnet
            if "l1_ratio" not in kw:
                kw["l1_ratio"] = 0.5  # Default l1_ratio for elasticnet (must be in [0, 1])
    
    # Sanitize parameters
    kw = safe_kwargs_for(cls, kw)
    
    return cls(**kw)


def build_vote(pos_nodes: list, kw_nodes: dict) -> Any:
    """
    Build a VotingClassifier from AST nodes.
    
    Args:
        pos_nodes: List of positional argument nodes (base estimators)
        kw_nodes: Dictionary of keyword argument nodes (ensemble options)
        
    Returns:
        Compiled VotingClassifier
    """
    # Get voting mode (default to "hard")
    voting_node = kw_nodes.get("voting", {"type": "literal", "value": "hard"})
    voting = eval_value(voting_node)
    
    # Create context for child models
    context = {"voting": voting}
    
    # Compile base estimators
    estimators = []
    for i, child_node in enumerate(pos_nodes):
        est = compile_ast_to_estimator(child_node, context=context)
        estimators.append((f"m{i}", est))
    
    # Get other options (weights, etc.)
    options = {}
    for k, v in kw_nodes.items():
        if k != "voting":  # Already handled
            options[k] = eval_value(v)
    
    # Sanitize options
    options = safe_kwargs_for(VotingClassifier, options)
    options["voting"] = voting
    
    return VotingClassifier(estimators=estimators, **options)


def build_stack(pos_nodes: list, kw_nodes: dict, context: dict = None) -> Any:
    """
    Build a StackingClassifier from AST nodes.
    
    Args:
        pos_nodes: List of positional argument nodes (base estimators)
        kw_nodes: Dictionary of keyword argument nodes (ensemble options)
        context: Context dictionary
        
    Returns:
        Compiled StackingClassifier
    """
    if context is None:
        context = {}
    
    # Compile base estimators
    estimators = []
    for i, child_node in enumerate(pos_nodes):
        est = compile_ast_to_estimator(child_node, context)
        estimators.append((f"m{i}", est))
    
    # Get options
    options = {}
    for k, v in kw_nodes.items():
        if k == "final_estimator":
            # final_estimator should be a model name string, compile it
            model_name = eval_value(v)
            if model_name in MODEL_REGISTRY:
                options[k] = MODEL_REGISTRY[model_name]()
            else:
                options[k] = model_name
        else:
            options[k] = eval_value(v)
    
    # Sanitize options
    options = safe_kwargs_for(StackingClassifier, options)
    
    # Default final_estimator if not provided
    if "final_estimator" not in options:
        options["final_estimator"] = LogisticRegression()
    
    return StackingClassifier(estimators=estimators, **options)


def build_bag(pos_nodes: list, kw_nodes: dict, context: dict = None) -> Any:
    """
    Build a BaggingClassifier from AST nodes.
    
    Args:
        pos_nodes: List with single base estimator node
        kw_nodes: Dictionary of keyword argument nodes
        context: Context dictionary
        
    Returns:
        Compiled BaggingClassifier
    """
    if context is None:
        context = {}
    
    if len(pos_nodes) != 1:
        raise ValueError("BaggingClassifier requires exactly one base estimator")
    
    # Compile the base estimator
    base_estimator = compile_ast_to_estimator(pos_nodes[0], context)
    
    # Get options
    options = {k: eval_value(v) for k, v in kw_nodes.items()}
    options = safe_kwargs_for(BaggingClassifier, options)
    
    return BaggingClassifier(estimator=base_estimator, **options)


def build_ada(pos_nodes: list, kw_nodes: dict, context: dict = None) -> Any:
    """
    Build an AdaBoostClassifier from AST nodes.
    
    Args:
        pos_nodes: List with single base estimator node
        kw_nodes: Dictionary of keyword argument nodes
        context: Context dictionary
        
    Returns:
        Compiled AdaBoostClassifier
    """
    if context is None:
        context = {}
    
    if len(pos_nodes) != 1:
        raise ValueError("AdaBoostClassifier requires exactly one base estimator")
    
    # Compile the base estimator
    base_estimator = compile_ast_to_estimator(pos_nodes[0], context)
    
    # Get options
    options = {k: eval_value(v) for k, v in kw_nodes.items()}
    options = safe_kwargs_for(AdaBoostClassifier, options)
    
    # Default n_estimators if not provided
    if "n_estimators" not in options:
        options["n_estimators"] = 50
    
    return AdaBoostClassifier(estimator=base_estimator, **options)


def compile_ast_to_estimator(node: dict, context: dict = None) -> Any:
    """
    Compile a dict-based AST node into a scikit-learn estimator.
    
    Args:
        node: AST node (dict with "type" key)
        context: Context dictionary for passing information (e.g., voting mode)
        
    Returns:
        A scikit-learn estimator
    """
    if context is None:
        context = {}
    
    if node["type"] == "literal":
        # Literals at top-level are not valid (should be a call)
        raise ValueError("Top-level AST node cannot be a literal")
    
    if node["type"] != "call":
        raise ValueError(f"Expected call node, got: {node.get('type')}")
    
    name = node["name"]
    
    # Check if it's a model
    if name in MODEL_REGISTRY:
        return build_model(name, node["kw"], context)
    
    # Check if it's an ensemble
    if name == "vote":
        return build_vote(node["pos"], node["kw"])
    
    if name == "stack":
        return build_stack(node["pos"], node["kw"], context)
    
    if name == "bag":
        return build_bag(node["pos"], node["kw"], context)
    
    if name == "ada":
        return build_ada(node["pos"], node["kw"], context)
    
    raise ValueError(f"Unknown function/model: {name}")


def dsl_to_sklearn_estimator(dsl_string: str):
    """
    Convert DSL string directly to sklearn estimator.
    
    This is the main entry point: DSL string → sklearn estimator.
    
    Args:
        dsl_string: DSL string to compile
        
    Returns:
        Compiled sklearn estimator (not fitted)
    """
    from Torque_mapper import map_dsl_to_ast as parse_dsl_to_ast
    
    ast = parse_dsl_to_ast(dsl_string)
    est = compile_ast_to_estimator(ast)
    return est
