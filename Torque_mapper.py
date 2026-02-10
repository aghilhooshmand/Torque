"""
Torque Mapper - Converts Torque DSL Commands to Scikit-Learn Python Code

This module provides a clean interface to convert Torque DSL strings into
runnable Python code that uses scikit-learn.

Responsibilities:
- Parse Torque DSL to AST (embedded pyparsing-based DSL mapper)
- Convert AST to Python code
- Export AST and Python code to JSON file

This module does NOT execute code - that's handled by Torque_runner.py.

Example:
    Input:  'vote(LR(C=1.0), SVM(kernel="rbf"); voting="hard")'
    Output: Python code that creates a VotingClassifier with LogisticRegression
            and SVC as base estimators.

Usage:
    mapper = TorqueMapper()
    # Convert DSL to AST
    ast = mapper.dsl_to_ast('vote(LR(C=1.0), SVM(kernel="rbf"))')
    # Convert AST to Python code
    python_code = mapper.ast_to_python(ast)
    # Or export everything to JSON
    mapper.export_to_json('vote(LR(C=1.0), SVM())', 'Torque_mapper_result.json')
"""

from typing import Dict, List, Any, Tuple, Optional
import sys
import os
import json
from datetime import datetime

from pyparsing import (
    Forward,
    Group,
    Keyword,
    Optional as PPOptional,
    Suppress,
    Word,
    alphanums,
    alphas,
    delimitedList,
    pyparsing_common,
    quotedString,
    NotAny,
    ParseException,
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from registry import MODEL_REGISTRY, ENSEMBLE_REGISTRY


# ---------------------------------------------------------------------------
# Embedded DSL → AST mapper (previously in dsl_mapper.py)
# ---------------------------------------------------------------------------

class DSLMapper:
    """Mapper for Torque DSL using pyparsing - maps DSL commands to AST."""

    def __init__(self) -> None:
        self._setup_grammar()

    def _setup_grammar(self) -> None:
        """Set up the pyparsing grammar for mapping DSL."""
        # 1. Basic tokens
        IDENT = Word(alphas + "_", alphanums + "_")
        NUMBER = pyparsing_common.number()
        STRING = quotedString.setParseAction(lambda t: t[0][1:-1])  # Remove quotes

        # Boolean and null literals
        TRUE = Keyword("True").setParseAction(lambda: True)
        FALSE = Keyword("False").setParseAction(lambda: False)
        NULL = Keyword("None").setParseAction(lambda: None)

        # 2. Recursive expression (for nested calls) - must be Forward
        EXPR = Forward()

        # 3. Keyword argument: key=value
        KWARG = Group(IDENT("key") + Suppress("=") + EXPR("value"))

        # 4. Argument lists
        POSARGS_LIST = delimitedList(EXPR)("pos")
        KWARGS_LIST = delimitedList(KWARG)("kw")

        # 5. Function call patterns - try most specific first
        # Pattern 1: name(posargs ; kwargs) - ensemble style
        CALL_PATTERN1 = Group(
            IDENT("name")
            + Suppress("(")
            + POSARGS_LIST
            + Suppress(";")
            + KWARGS_LIST
            + Suppress(")")
        ).setParseAction(self._make_call_node)

        # Pattern 2: name(kwargs) - model style with only keyword args
        CALL_PATTERN2 = Group(
            IDENT("name")
            + Suppress("(")
            + KWARGS_LIST
            + Suppress(")")
        ).setParseAction(self._make_call_node)

        # Pattern 3: name(posargs) - only positional args
        CALL_PATTERN3 = Group(
            IDENT("name")
            + Suppress("(")
            + POSARGS_LIST
            + Suppress(")")
        ).setParseAction(self._make_call_node)

        # Pattern 4: name() - empty
        CALL_PATTERN4 = Group(
            IDENT("name")
            + Suppress("(")
            + Suppress(")")
        ).setParseAction(self._make_call_node)

        # Combine patterns - try in order of specificity
        CALL = CALL_PATTERN1 | CALL_PATTERN2 | CALL_PATTERN3 | CALL_PATTERN4

        # 6. Atomic values (literals only)
        ATOM = STRING | NUMBER | TRUE | FALSE | NULL

        # 7. Bare identifier (only if NOT followed by '(' - to avoid conflict with CALL)
        BARE_IDENT = IDENT + NotAny(Suppress("("))
        BARE_IDENT.setParseAction(self._make_literal_node)

        # 8. Expression can be a call, atomic value, or bare identifier
        # IMPORTANT: CALL must come first to handle nested calls like LR(C=1.0)
        EXPR <<= (CALL | ATOM.setParseAction(self._make_literal_node) | BARE_IDENT)

        self.grammar = EXPR

    def _make_literal_node(self, tokens):
        """Create a dict-based LiteralNode from mapped tokens."""
        value = tokens[0]
        return {
            "type": "literal",
            "value": value,
        }

    def _make_call_node(self, tokens):
        """Create a dict-based CallNode from mapped tokens."""
        if len(tokens) == 0:
            return {"type": "call", "name": "", "pos": [], "kw": {}}

        token_data = tokens[0]

        # Name
        if hasattr(token_data, "name"):
            name = token_data.name
        elif len(token_data) > 0:
            name = str(token_data[0])
        else:
            name = ""

        # Positional arguments
        pos: List[Any] = []
        if hasattr(token_data, "pos") and token_data.pos:
            pos = [item for item in token_data.pos]

        # Keyword arguments
        kw: Dict[str, Any] = {}
        if hasattr(token_data, "kw") and token_data.kw:
            for kwarg in token_data.kw:
                if hasattr(kwarg, "key") and hasattr(kwarg, "value"):
                    key = kwarg.key
                    val = kwarg.value
                    if hasattr(val, "__len__") and len(val) == 1 and not isinstance(val, str):
                        value = val[0]
                    else:
                        value = val
                    kw[key] = value

        return {
            "type": "call",
            "name": name,
            "pos": pos,
            "kw": kw,
        }

    def map(self, dsl_string: str) -> dict:
        """
        Map a DSL string into an AST.
        
        Args:
            dsl_string: The Torque DSL command to map
        
        Returns:
            dict representing the root of the AST
        """
        try:
            result = self.grammar.parseString(dsl_string, parseAll=True)
        except ParseException as e:
            # Build a helpful syntax error message with line/column and pointer
            line = e.line or dsl_string
            col = e.col or 1
            pointer = " " * (col - 1) + "^"
            # e.msg is a short description like "Expected ')'"
            msg = f"Syntax error at column {col}: {e.msg}"
            raise ValueError(f"{msg}\n\n{line}\n{pointer}") from e
        
        if not result:
            raise ValueError(f"Failed to map DSL (empty parse result).")
        return result[0]


_DSL_MAPPER: Optional[DSLMapper] = None


def map_dsl_to_ast(dsl_string: str) -> dict:
    """
    Map a Torque DSL command into a dict-based AST.

    Args:
        dsl_string: The Torque DSL command to map

    Returns:
        dict representing the root of the AST (CallNode or LiteralNode structure)
    """
    global _DSL_MAPPER
    if _DSL_MAPPER is None:
        _DSL_MAPPER = DSLMapper()
    return _DSL_MAPPER.map(dsl_string)


class TorqueMapper:
    """
    Maps Torque DSL commands to AST and Python code.
    
    This class converts Torque DSL strings (like 'vote(LR(C=1.0), SVM())')
    into AST (Abstract Syntax Tree) and then into Python code that can be 
    executed to create scikit-learn estimators.
    
    This class does NOT execute code - it only converts DSL to code.
    Use runner.py to execute the generated Python code.
    
    Attributes:
        model_registry: Dictionary mapping DSL model names to sklearn classes
        ensemble_registry: Dictionary mapping DSL ensemble names to sklearn classes
    """
    
    def __init__(self):
        """Initialize the mapper with model and ensemble registries."""
        self.model_registry = MODEL_REGISTRY
        self.ensemble_registry = ENSEMBLE_REGISTRY
    
    def dsl_to_ast(self, torque_command: str) -> Dict:
        """
        Convert Torque DSL command to AST (Abstract Syntax Tree).
        
        Args:
            torque_command: The Torque DSL string (e.g., 'vote(LR(C=1.0), SVM())')
            
        Returns:
            Dictionary representing the AST structure
            
        Example:
            >>> mapper = TorqueMapper()
            >>> ast = mapper.dsl_to_ast('vote(LR(C=1.0), SVM())')
            >>> print(ast['name'])  # 'vote'
        """
        return map_dsl_to_ast(torque_command)
    
    def ast_to_python(self, ast: Dict, variable_name: str = "estimator") -> str:
        """
        Convert AST to runnable Python code string.
        
        Args:
            ast: AST dictionary from dsl_to_ast()
            variable_name: Name of the variable to store the result (default: "estimator")
            
        Returns:
            Complete Python code string with imports and estimator creation
            
        Example:
            >>> mapper = TorqueMapper()
            >>> ast = mapper.dsl_to_ast('vote(LR(C=1.0), SVM())')
            >>> code = mapper.ast_to_python(ast)
        """
        # Generate Python code from AST
        python_code = self._ast_to_python(ast, variable_name)
        
        # Add necessary imports at the top
        imports = self._generate_imports(ast)
        
        # Combine imports and code
        full_code = imports + "\n\n" + python_code
        
        return full_code
    
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
            String containing import statements
        """
        # For simplicity, import all commonly used classes
        # In a production system, you might want to analyze AST and import only what's needed
        import_lines = [
            "from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.tree import DecisionTreeClassifier",
            "from sklearn.naive_bayes import GaussianNB"
        ]
        
        return "\n".join(import_lines)
    
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
    
    def export_to_json(
        self,
        torque_command: str,
        output_file: str = "Torque_mapper_result.json",
        variable_name: str = "estimator"
    ) -> Dict[str, Any]:
        """
        Convert Torque DSL command to AST and Python code, then export to JSON file.
        
        This method:
        1. Converts DSL to AST
        2. Converts AST to Python code
        3. Saves both AST and Python code to JSON file
        
        Args:
            torque_command: Torque DSL string (e.g., 'vote(LR(C=1.0), SVM())')
            output_file: Path to output JSON file (default: "torque_mapped.json")
            variable_name: Name of the variable in generated Python code (default: "estimator")
            
        Returns:
            Dictionary containing DSL, AST, Python code, and metadata
            
        Example:
            >>> mapper = TorqueMapper()
            >>> result = mapper.export_to_json(
            ...     'vote(LR(C=1.0), SVM())',
            ...     output_file='mapped_code.json'
            ... )
            >>> print(f"Python code saved to: {result['output_file']}")
        """
        # Step 1: Convert DSL to AST
        ast = self.dsl_to_ast(torque_command)
        
        # Step 2: Convert AST to Python code
        python_code = self.ast_to_python(ast, variable_name)
        
        # Step 3: Prepare output dictionary
        output_data = {
            "torque_command": torque_command,
            "timestamp": datetime.now().isoformat(),
            "variable_name": variable_name,
            "ast": ast,
            "python_code": python_code
        }
        
        # Step 4: Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        output_data["output_file"] = output_file
        
        return output_data


# Convenience function
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


# Command-line interface
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Torque Mapper - Convert Torque DSL commands to Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All settings can be read from a config file, or specified via command-line.

Examples:
  # Using config file (recommended)
  python Torque_mapper.py --config config/Torque_mapper_config.json
  
  # Using command-line arguments
  python Torque_Mapper.py 'vote(LR(C=1.0), SVM())' -o mapped.json
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (if provided, all other arguments are ignored)"
    )
    
    parser.add_argument(
        "torque_command",
        nargs="?",
        type=str,
        default=None,
        help="Torque DSL command (e.g., 'vote(LR(C=1.0), SVM())') - required if --config not used"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated from command hash)"
    )
    
    parser.add_argument(
        "-v", "--variable",
        type=str,
        default="estimator",
        help="Variable name in generated Python code (default: 'estimator')"
    )
    
    parser.add_argument(
        "--print-code",
        action="store_true",
        help="Print generated Python code to stdout"
    )
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ Error: Config file not found: {args.config}")
            sys.exit(1)
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        torque_command = config.get("torque_command")
        if not torque_command:
            print("❌ Error: Config file must contain 'torque_command'")
            sys.exit(1)
        
        output_config = config.get("output", {})
        output_file = output_config.get("mapped_json_file")
        if not output_file:
            # Auto-generate if not specified
            import hashlib
            cmd_hash = hashlib.md5(torque_command.encode()).hexdigest()[:8]
            output_file = f"torque_mapped_{cmd_hash}.json"
        
        variable_name = config.get("variable_name", "estimator")
        print_code = config.get("print_code", False)
        
        print(f"✓ Loaded config from: {args.config}")
    else:
        # Use command-line arguments
        if not args.torque_command:
            parser.error("Either --config or torque_command must be provided")
        
        torque_command = args.torque_command
        output_file = args.output
        variable_name = args.variable
        print_code = args.print_code
        
        # Generate output filename if not provided
        if output_file is None:
            import hashlib
            cmd_hash = hashlib.md5(torque_command.encode()).hexdigest()[:8]
            output_file = f"torque_mapped_{cmd_hash}.json"
    
    # Create mapper
    mapper = TorqueMapper()
    
    # Export to JSON
    print(f"Converting Torque DSL to Python code...")
    print(f"  Command: {torque_command}")
    
    result = mapper.export_to_json(
        torque_command=torque_command,
        output_file=output_file,
        variable_name=variable_name
    )
    
    print(f"✓ Mapping saved to: {result['output_file']}")
    
    # Print code if requested
    if print_code:
        print("\n" + "=" * 70)
        print("Generated Python Code:")
        print("=" * 70)
        print(result['python_code'])
        print("=" * 70)
    
    print(f"\nUse Torque_runner.py to execute the Python code:")
    print(f"  python Torque_runner.py --config config/Torque_runner_config.json")
