"""
AST (Abstract Syntax Tree) node definitions for the DSL.

AST nodes are plain Python structures that represent the parsed DSL.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class CallNode(ASTNode):
    """Represents a function call in the DSL.
    
    Examples:
        - vote(LR(C=1.0), SVM(kernel="rbf"); voting="hard")
        - LR(C=1.0, penalty="l2")
    """
    name: str  # Function/model name (e.g., "vote", "LR", "SVM")
    pos: List[Any]  # Positional arguments (can be CallNodes or literals)
    kw: Dict[str, Any]  # Keyword arguments (named parameters)
    
    def __repr__(self) -> str:
        pos_str = ", ".join(str(p) for p in self.pos) if self.pos else ""
        kw_str = ", ".join(f"{k}={v}" for k, v in self.kw.items()) if self.kw else ""
        
        if pos_str and kw_str:
            return f"{self.name}({pos_str}; {kw_str})"
        elif pos_str:
            return f"{self.name}({pos_str})"
        elif kw_str:
            return f"{self.name}({kw_str})"
        else:
            return f"{self.name}()"


@dataclass
class LiteralNode(ASTNode):
    """Represents a literal value (string, number, bool, None)."""
    value: Any
    type: str  # "string", "number", "bool", "null"
    
    def __repr__(self) -> str:
        return str(self.value)

