"""Simple test to debug parser."""

from dsl_parser import parse_dsl

# Test simple cases
tests = [
    'LR(C=1.0)',
    'vote(LR(C=1.0); voting="hard")',
    'vote(LR(C=1.0), SVM(C=1.0); voting="hard")',
]

for test in tests:
    print(f"\nTesting: {test}")
    try:
        ast = parse_dsl(test)
        print(f"  ✅ Success: {ast}")
        print(f"  Type: {type(ast)}")
        if hasattr(ast, 'pos'):
            print(f"  Pos args: {ast.pos}")
        if hasattr(ast, 'kw'):
            print(f"  Kw args: {ast.kw}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

