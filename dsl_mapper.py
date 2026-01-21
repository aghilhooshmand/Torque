"""
DSL Mapper for Torque DSL.

Maps Torque DSL commands to AST nodes for scikit-learn.
"""

from typing import Any, List

from pyparsing import (
    Forward,
    Group,
    Keyword,
    Optional,
    Suppress,
    Word,
    alphanums,
    alphas,
    delimitedList,
    pyparsing_common,
    quotedString,
    NotAny,
)

# Import AST nodes
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# AST nodes are now dict-based for grammatical evolution compatibility
# No need to import dataclass-based nodes


class DSLMapper:
    """Mapper for Torque DSL using pyparsing - maps DSL commands to AST."""
    
    def __init__(self):
        self._setup_grammar()
    
    def _setup_grammar(self):
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
        
        # Return dict-based AST node (for grammatical evolution compatibility)
        return {
            "type": "literal",
            "value": value
        }
    
    def _make_call_node(self, tokens):
        """Create a dict-based CallNode from mapped tokens."""
        # tokens is a ParseResults list containing one Group result
        # The actual data is in tokens[0]
        if len(tokens) == 0:
            return {"type": "call", "name": "", "pos": [], "kw": {}}
        
        # Get the actual token data (the Group result)
        token_data = tokens[0]
        
        # Get name
        name = ""
        if hasattr(token_data, "name"):
            name = token_data.name
        elif len(token_data) > 0:
            name = str(token_data[0])
        
        # Positional arguments - convert to dict-based nodes
        pos = []
        if hasattr(token_data, "pos") and token_data.pos:
            pos = [item for item in token_data.pos]
        
        # Keyword arguments - convert to dict-based nodes
        kw = {}
        if hasattr(token_data, "kw") and token_data.kw:
            # token_data.kw is a ParseResults list of KWARG Groups
            for kwarg in token_data.kw:
                # kwarg is a ParseResults from Group(IDENT("key") + ... + EXPR("value"))
                if hasattr(kwarg, "key") and hasattr(kwarg, "value"):
                    key = kwarg.key
                    # Get value - handle ParseResults
                    val = kwarg.value
                    # If it's a ParseResults with one element, get that element
                    if hasattr(val, "__len__") and len(val) == 1 and not isinstance(val, str):
                        value = val[0]
                    else:
                        value = val
                    kw[key] = value
        
        # Return dict-based AST node (for grammatical evolution compatibility)
        return {
            "type": "call",
            "name": name,
            "pos": pos,
            "kw": kw
        }
    
    def map(self, dsl_string: str) -> dict:
        """
        Map a DSL string into an AST.
        
        Args:
            dsl_string: The Torque DSL command to map
            
        Returns:
            dict representing the root of the AST
            
        Raises:
            Exception: If mapping fails
        """
        result = self.grammar.parseString(dsl_string, parseAll=True)
        if not result:
            raise ValueError(f"Failed to map DSL: {dsl_string}")
        
        return result[0]


# Alias for backward compatibility
def map_dsl(dsl_string: str) -> dict:
    """Alias for map_dsl_to_ast (backward compatibility)."""
    return map_dsl_to_ast(dsl_string)


# Global mapper instance
_mapper = None


def map_dsl_to_ast(dsl_string: str) -> dict:
    """
    Map a Torque DSL command into a dict-based AST.
    
    Args:
        dsl_string: The Torque DSL command to map
        
    Returns:
        dict representing the root of the AST (CallNode or LiteralNode structure)
    """
    global _mapper
    if _mapper is None:
        _mapper = DSLMapper()
    
    return _mapper.map(dsl_string)


# Backward compatibility aliases
def parse_dsl_to_ast(dsl_string: str) -> dict:
    """Backward compatibility alias - use map_dsl_to_ast instead."""
    return map_dsl_to_ast(dsl_string)

def translate_dsl_to_ast(dsl_string: str) -> dict:
    """Backward compatibility alias - use map_dsl_to_ast instead."""
    return map_dsl_to_ast(dsl_string)
