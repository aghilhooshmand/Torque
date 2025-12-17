"""Test with debug output."""

# Temporarily modify the parser to add debug
import dsl_parser
original_make_call = dsl_parser.DSLParser._make_call_node

def debug_make_call(self, tokens):
    print(f"\n=== DEBUG ===")
    print(f"tokens: {tokens}")
    print(f"tokens type: {type(tokens)}")
    print(f"tokens list: {list(tokens)}")
    print(f"tokens len: {len(tokens)}")
    print(f"has name attr: {hasattr(tokens, 'name')}")
    print(f"has kw attr: {hasattr(tokens, 'kw')}")
    if hasattr(tokens, 'name'):
        print(f"tokens.name: {tokens.name}")
    if hasattr(tokens, 'kw'):
        print(f"tokens.kw: {tokens.kw}")
    if hasattr(tokens, 'pos'):
        print(f"tokens.pos: {tokens.pos}")
    print(f"=============\n")
    return original_make_call(self, tokens)

dsl_parser.DSLParser._make_call_node = debug_make_call

# Now test
from dsl_parser import parse_dsl

test = 'LR(C=1.0)'
print(f"Testing: {test}")
ast = parse_dsl(test)
print(f"Result - Name: '{ast.name}', Kw: {ast.kw}")

