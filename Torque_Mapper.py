"""
Torque Mapper - Converts Torque DSL Commands to Scikit-Learn Python Code

This module provides a clean interface to convert Torque DSL strings into
runnable Python code that uses scikit-learn.

Example:
    Input:  'vote(LR(C=1.0), SVM(kernel="rbf"); voting="hard")'
    Output: Python code that creates a VotingClassifier with LogisticRegression
            and SVC as base estimators.

Usage:
    mapper = TorqueMapper()
    python_code = mapper.map_to_python('vote(LR(C=1.0), SVM(kernel="rbf"))')
    exec(python_code)  # Execute the generated code
"""

from typing import Dict, List, Any
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dsl_mapper import map_dsl_to_ast
from registry import MODEL_REGISTRY, ENSEMBLE_REGISTRY


class TorqueMapper:
    """
    Maps Torque DSL commands to runnable scikit-learn Python code.
    
    This class converts Torque DSL strings (like 'vote(LR(C=1.0), SVM())')
    into Python code that can be executed to create scikit-learn estimators.
    
    Attributes:
        model_registry: Dictionary mapping DSL model names to sklearn classes
        ensemble_registry: Dictionary mapping DSL ensemble names to sklearn classes
    """
    
    def __init__(self):
        """Initialize the mapper with model and ensemble registries."""
        self.model_registry = MODEL_REGISTRY
        self.ensemble_registry = ENSEMBLE_REGISTRY
    
    def map_to_python(self, torque_command: str, variable_name: str = "estimator") -> str:
        """
        Convert a Torque DSL command to runnable Python code.
        
        Args:
            torque_command: The Torque DSL string (e.g., 'vote(LR(C=1.0), SVM())')
            variable_name: Name of the variable to store the result (default: "estimator")
            
        Returns:
            Complete Python code string with imports and estimator creation
            
        Example:
            >>> mapper = TorqueMapper()
            >>> code = mapper.map_to_python('vote(LR(C=1.0), SVM())')
            >>> print(code)
            # Generated Python code with imports and estimator
        """
        # Parse the Torque command into AST
        ast = map_dsl_to_ast(torque_command)
        
        # Generate Python code from AST
        python_code = self._ast_to_python(ast, variable_name)
        
        # Add necessary imports at the top
        imports = self._generate_imports(ast)
        
        # Combine imports and code
        full_code = imports + "\n\n" + python_code
        
        return full_code
    
    def _generate_imports(self, ast: Dict) -> str:
        """
        Generate necessary scikit-learn imports based on AST content.
        
        Args:
            ast: The AST dictionary representing the Torque command
            
        Returns:
            String containing all necessary import statements
        """
        imports = set()
        
        # Traverse AST to find all used models and ensembles
        self._collect_imports(ast, imports)
        
        # Build import statements
        import_lines = [
            "from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.svm import SVC",
            "from sklearn.tree import DecisionTreeClassifier, RandomForestClassifier",
            "from sklearn.naive_bayes import GaussianNB"
        ]
        
        return "\n".join(import_lines)
    
    def _collect_imports(self, node: Dict, imports: set):
        """
        Recursively collect all model and ensemble names from AST.
        
        Args:
            node: AST node (call or literal)
            imports: Set to collect import names into
        """
        if node.get("type") == "call":
            name = node.get("name", "")
            # Check if it's a model or ensemble
            if name in self.model_registry:
                imports.add(f"model_{name}")
            elif name in self.ensemble_registry:
                imports.add(f"ensemble_{name}")
            
            # Recursively process positional and keyword arguments
            for pos_arg in node.get("pos", []):
                self._collect_imports(pos_arg, imports)
            
            for kw_value in node.get("kw", {}).values():
                self._collect_imports(kw_value, imports)
    
    def _ast_to_python(self, ast: Dict, variable_name: str, indent: int = 0) -> str:
        """
        Convert AST node to Python code string.
        
        Args:
            ast: AST dictionary node
            variable_name: Name for the resulting variable
            indent: Current indentation level (for nested structures)
            
        Returns:
            Python code string
        """
        indent_str = "    " * indent
        
        if ast.get("type") == "literal":
            # Literal value - return as Python literal
            value = ast.get("value")
            return self._python_literal(value)
        
        elif ast.get("type") == "call":
            name = ast.get("name", "")
            pos_args = ast.get("pos", [])
            kw_args = ast.get("kw", {})
            
            # Check if it's a model (LR, SVM, RF, DT, NB)
            if name in self.model_registry:
                return self._build_model_code(name, kw_args, variable_name, indent)
            
            # Check if it's an ensemble (vote, stack, bag, ada)
            elif name in self.ensemble_registry:
                return self._build_ensemble_code(name, pos_args, kw_args, variable_name, indent)
            
            else:
                raise ValueError(f"Unknown model or ensemble: {name}")
        
        else:
            raise ValueError(f"Unknown AST node type: {ast.get('type')}")
    
    def _build_model_code(self, model_name: str, kw_args: Dict, var_name: str, indent: int) -> str:
        """
        Build Python code for a scikit-learn model.
        
        Args:
            model_name: Model name (e.g., "LR", "SVM")
            kw_args: Dictionary of keyword arguments (AST nodes)
            var_name: Variable name for the result
            indent: Indentation level
            
        Returns:
            Python code string
        """
        indent_str = "    " * indent
        sklearn_class = self.model_registry[model_name].__name__
        
        # Convert keyword arguments from AST to Python values
        python_kwargs = {}
        for key, value_node in kw_args.items():
            python_kwargs[key] = self._ast_to_python(value_node, "", indent + 1)
        
        # Build parameter string
        if python_kwargs:
            params = ", ".join(f"{k}={v}" for k, v in python_kwargs.items())
            code = f"{indent_str}{var_name} = {sklearn_class}({params})"
        else:
            code = f"{indent_str}{var_name} = {sklearn_class}()"
        
        return code
    
    def _build_ensemble_code(self, ensemble_name: str, pos_args: List, kw_args: Dict, 
                             var_name: str, indent: int) -> str:
        """
        Build Python code for a scikit-learn ensemble.
        
        Args:
            ensemble_name: Ensemble name (e.g., "vote", "stack", "bag", "ada")
            pos_args: List of positional arguments (base estimators)
            kw_args: Dictionary of keyword arguments (ensemble options)
            var_name: Variable name for the result
            indent: Indentation level
            
        Returns:
            Python code string
        """
        indent_str = "    " * indent
        sklearn_class = self.ensemble_registry[ensemble_name].__name__
        
        lines = []
        
        # Build base estimators list for ensembles
        if pos_args:
            # Create temporary variables for base estimators
            base_estimators = []
            for i, pos_arg in enumerate(pos_args):
                base_var = f"base_estimator_{i}"
                base_code = self._ast_to_python(pos_arg, base_var, indent)
                lines.append(base_code)
                base_estimators.append(base_var)
            
            # Build estimators parameter based on ensemble type
            if ensemble_name in ["vote", "stack"]:
                # VotingClassifier and StackingClassifier use 'estimators' parameter
                # Format: [('name0', estimator0), ('name1', estimator1), ...]
                estimators_tuples = [f"('{name}', {var})" for name, var in 
                                    zip([f"model_{i}" for i in range(len(base_estimators))], base_estimators)]
                estimators_list = f"[{', '.join(estimators_tuples)}]"
                estimators_param = f"estimators={estimators_list}"
            else:
                # BaggingClassifier and AdaBoostClassifier use 'base_estimator' (singular)
                if len(pos_args) > 1:
                    raise ValueError(f"{ensemble_name} only accepts one base estimator, got {len(pos_args)}")
                estimators_param = f"base_estimator={base_estimators[0]}"
        else:
            estimators_param = ""
        
        # Convert keyword arguments
        python_kwargs = {}
        for key, value_node in kw_args.items():
            python_kwargs[key] = self._ast_to_python(value_node, "", indent + 1)
        
        # Combine all parameters
        all_params = []
        if estimators_param:
            all_params.append(estimators_param)
        all_params.extend(f"{k}={v}" for k, v in python_kwargs.items())
        
        # Build the ensemble creation line
        if all_params:
            params_str = ", ".join(all_params)
            ensemble_line = f"{indent_str}{var_name} = {sklearn_class}({params_str})"
        else:
            ensemble_line = f"{indent_str}{var_name} = {sklearn_class}()"
        
        lines.append(ensemble_line)
        
        return "\n".join(lines)
    
    def _python_literal(self, value: Any) -> str:
        """
        Convert a Python value to its string representation.
        
        Args:
            value: Python value (str, int, float, bool, None)
            
        Returns:
            String representation suitable for Python code
        """
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        else:
            return str(value)


def map_torque_to_python(torque_command: str, variable_name: str = "estimator") -> str:
    """
    Convenience function to map Torque DSL command to Python code.
    
    Args:
        torque_command: The Torque DSL string
        variable_name: Name of the variable to store the result
        
    Returns:
        Complete Python code string
        
    Example:
        >>> code = map_torque_to_python('vote(LR(C=1.0), SVM())')
        >>> print(code)
    """
    mapper = TorqueMapper()
    return mapper.map_to_python(torque_command, variable_name)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Simple model
    print("=" * 60)
    print("Example 1: Simple Logistic Regression")
    print("=" * 60)
    mapper = TorqueMapper()
    code1 = mapper.map_to_python('LR(C=1.0, max_iter=100)')
    print(code1)
    print()
    
    # Example 2: Voting ensemble
    print("=" * 60)
    print("Example 2: Voting Classifier")
    print("=" * 60)
    code2 = mapper.map_to_python('vote(LR(C=1.0), SVM(kernel="rbf"); voting="hard")')
    print(code2)
    print()
    
    # Example 3: Stacking ensemble
    print("=" * 60)
    print("Example 3: Stacking Classifier")
    print("=" * 60)
    code3 = mapper.map_to_python('stack(LR(C=1.0), SVM(), RF(n_estimators=100))')
    print(code3)
