"""Minimal test to debug KWARG parsing."""

from pyparsing import Group, Suppress, Word, alphas, alphanums, delimitedList, pyparsing_common, Forward

IDENT = Word(alphas + "_", alphanums + "_")
NUMBER = pyparsing_common.number()

EXPR = Forward()
KWARG = Group(IDENT("key") + Suppress("=") + EXPR("value"))
KWARGS_LIST = delimitedList(KWARG)("kw")
CALL = Group(IDENT("name") + Suppress("(") + KWARGS_LIST + Suppress(")"))

# Define EXPR - just NUMBER for now
EXPR <<= NUMBER

# Test
test = "LR(C=1.0)"
print(f"Testing: {test}")
result = CALL.parseString(test, parseAll=True)
print(f"Result: {result}")
print(f"Result[0]: {result[0]}")
print(f"Has kw: {hasattr(result[0], 'kw')}")
if hasattr(result[0], 'kw'):
    print(f"kw: {result[0].kw}")
    if result[0].kw:
        print(f"kw[0]: {result[0].kw[0]}")
        print(f"kw[0].key: {result[0].kw[0].key}")
        print(f"kw[0].value: {result[0].kw[0].value}")

