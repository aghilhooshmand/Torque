"""
Parser for Torque DSL - recursively parses nested function calls with Python-like syntax.

Example: func1(func2(param2=value),func3(param3="text"),param1=123)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Parameter:
    """Represents a function parameter (positional or keyword)."""
    
    name: Optional[str]  # None for positional, str for keyword
    value: Any  # Can be str, int, float, bool, None, list, dict, or FunctionNode
    
    def __repr__(self) -> str:
        if self.name:
            return f"{self.name}={self.value}"
        return str(self.value)


@dataclass
class FunctionNode:
    """Represents a function call node in the parse tree."""
    
    name: str
    parameters: List[Parameter]  # List of Parameter objects
    
    def __repr__(self) -> str:
        return f"FunctionNode(name='{self.name}', params={len(self.parameters)})"


class DSLParser:
    """Parser for Torque DSL function calls."""
    
    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0
        self.length = len(self.text)
    
    def parse(self) -> Optional[FunctionNode]:
        """
        Parse the DSL text and return the root function node.
        
        Returns:
            FunctionNode if parsing succeeds, None otherwise
        """
        if not self.text:
            return None
        
        try:
            self.pos = 0
            node = self._parse_function()
            return node
        except Exception as e:
            return None
    
    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1
    
    def _parse_function(self) -> FunctionNode:
        """Parse a function call: name(param1,param2,...)"""
        self._skip_whitespace()
        
        # Parse function name
        name_start = self.pos
        while (self.pos < self.length and 
               (self.text[self.pos].isalnum() or self.text[self.pos] in ['_', '-'])):
            self.pos += 1
        
        if self.pos == name_start:
            raise ValueError("Expected function name")
        
        name = self.text[name_start:self.pos]
        self._skip_whitespace()
        
        # Check for opening parenthesis
        if self.pos >= self.length or self.text[self.pos] != '(':
            # No parameters - just a name
            return FunctionNode(name=name, parameters=[])
        
        self.pos += 1  # Skip '('
        self._skip_whitespace()
        
        # Parse parameters
        parameters: List[Parameter] = []
        
        if self.pos < self.length and self.text[self.pos] == ')':
            # Empty parameter list
            self.pos += 1
            return FunctionNode(name=name, parameters=[])
        
        while self.pos < self.length:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            if self.text[self.pos] == ')':
                self.pos += 1
                break
            
            # Parse a parameter (could be keyword arg: name=value or positional)
            param = self._parse_parameter()
            if param is not None:
                parameters.append(param)
            
            self._skip_whitespace()
            
            # Check for comma or closing parenthesis
            if self.pos < self.length and self.text[self.pos] == ',':
                self.pos += 1
                continue
            elif self.pos < self.length and self.text[self.pos] == ')':
                self.pos += 1
                break
        
        return FunctionNode(name=name, parameters=parameters)
    
    def _parse_parameter(self) -> Optional[Parameter]:
        """Parse a parameter - could be keyword (name=value) or positional."""
        self._skip_whitespace()
        
        if self.pos >= self.length:
            return None
        
        start_pos = self.pos
        
        # Try to parse identifier (for keyword argument name)
        name_start = self.pos
        while (self.pos < self.length and 
               (self.text[self.pos].isalnum() or self.text[self.pos] in ['_'])):
            self.pos += 1
        
        param_name = None
        if self.pos > name_start:
            # We found an identifier
            identifier = self.text[name_start:self.pos]
            self._skip_whitespace()
            
            # Check if it's a keyword argument (name=value)
            if self.pos < self.length and self.text[self.pos] == '=':
                param_name = identifier
                self.pos += 1  # Skip '='
                self._skip_whitespace()
            else:
                # It's not a keyword arg, reset and parse as value
                self.pos = start_pos
        
        # Parse the value (could be function call, string, number, list, dict, etc.)
        value = self._parse_value()
        
        if value is None:
            return None
        
        return Parameter(name=param_name, value=value)
    
    def _parse_value(self) -> Optional[Any]:
        """Parse a value - function call, string, number, list, dict, bool, None."""
        self._skip_whitespace()
        
        if self.pos >= self.length:
            return None
        
        char = self.text[self.pos]
        
        # String literals
        if char in ['"', "'"]:
            return self._parse_string()
        
        # List literal
        if char == '[':
            return self._parse_list()
        
        # Dictionary literal
        if char == '{':
            return self._parse_dict()
        
        # Function call (identifier followed by '(')
        if char.isalnum() or char == '_':
            start_pos = self.pos
            while (self.pos < self.length and 
                   (self.text[self.pos].isalnum() or self.text[self.pos] in ['_'])):
                self.pos += 1
            
            identifier = self.text[start_pos:self.pos]
            self._skip_whitespace()
            
            if self.pos < self.length and self.text[self.pos] == '(':
                # It's a function call
                self.pos = start_pos
                return self._parse_function()
            
            # Check for boolean or None
            if identifier == 'True':
                return True
            if identifier == 'False':
                return False
            if identifier == 'None':
                return None
            
            # Try to parse as number
            self.pos = start_pos
            return self._parse_number_or_identifier()
        
        # Try to parse as number
        return self._parse_number_or_identifier()
    
    def _parse_string(self) -> str:
        """Parse a string literal (single or double quoted)."""
        quote = self.text[self.pos]
        self.pos += 1
        value_start = self.pos
        
        while self.pos < self.length:
            if self.text[self.pos] == '\\':
                self.pos += 2  # Skip escaped character
                continue
            if self.text[self.pos] == quote:
                value = self.text[value_start:self.pos]
                self.pos += 1  # Skip closing quote
                return value
            self.pos += 1
        
        raise ValueError("Unclosed string literal")
    
    def _parse_number_or_identifier(self) -> Union[int, float, str]:
        """Parse a number (int or float) or identifier."""
        start_pos = self.pos
        
        # Try to parse number
        if self.text[self.pos] == '-':
            self.pos += 1
        
        # Parse integer part
        while self.pos < self.length and self.text[self.pos].isdigit():
            self.pos += 1
        
        # Check for float
        if self.pos < self.length and self.text[self.pos] == '.':
            self.pos += 1
            while self.pos < self.length and self.text[self.pos].isdigit():
                self.pos += 1
            
            # Parse as float
            try:
                value = float(self.text[start_pos:self.pos])
                return value
            except ValueError:
                pass
        
        # Try to parse as integer
        if self.pos > start_pos:
            try:
                value = int(self.text[start_pos:self.pos])
                return value
            except ValueError:
                pass
        
        # Parse as identifier or unquoted string
        self.pos = start_pos
        value_start = self.pos
        while (self.pos < self.length and 
               self.text[self.pos] not in [',', ')', ']', '}', '='] and
               not self.text[self.pos].isspace()):
            self.pos += 1
        
        value = self.text[value_start:self.pos].strip()
        return value if value else None
    
    def _parse_list(self) -> List[Any]:
        """Parse a list literal: [item1, item2, ...]"""
        self.pos += 1  # Skip '['
        self._skip_whitespace()
        
        items = []
        
        if self.pos < self.length and self.text[self.pos] == ']':
            self.pos += 1
            return items
        
        while self.pos < self.length:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            if self.text[self.pos] == ']':
                self.pos += 1
                break
            
            # Parse list item
            item = self._parse_value()
            if item is not None:
                items.append(item)
            
            self._skip_whitespace()
            
            if self.pos < self.length and self.text[self.pos] == ',':
                self.pos += 1
                continue
            elif self.pos < self.length and self.text[self.pos] == ']':
                self.pos += 1
                break
        
        return items
    
    def _parse_dict(self) -> Dict[str, Any]:
        """Parse a dictionary literal: {key1: value1, key2: value2, ...}"""
        self.pos += 1  # Skip '{'
        self._skip_whitespace()
        
        items = {}
        
        if self.pos < self.length and self.text[self.pos] == '}':
            self.pos += 1
            return items
        
        while self.pos < self.length:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            if self.text[self.pos] == '}':
                self.pos += 1
                break
            
            # Parse key
            key = self._parse_value()
            if key is None:
                break
            
            self._skip_whitespace()
            
            if self.pos >= self.length or self.text[self.pos] != ':':
                break
            
            self.pos += 1  # Skip ':'
            self._skip_whitespace()
            
            # Parse value
            value = self._parse_value()
            if value is not None:
                items[str(key)] = value
            
            self._skip_whitespace()
            
            if self.pos < self.length and self.text[self.pos] == ',':
                self.pos += 1
                continue
            elif self.pos < self.length and self.text[self.pos] == '}':
                self.pos += 1
                break
        
        return items


def parse_dsl(text: str) -> Optional[FunctionNode]:
    """
    Parse DSL text and return the parse tree.
    
    Args:
        text: DSL text to parse
        
    Returns:
        FunctionNode representing the root of the parse tree, or None if parsing fails
    """
    parser = DSLParser(text)
    return parser.parse()

