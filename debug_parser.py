"""Debug parser to see token structure."""

from dsl_parser import DSLParser

parser = DSLParser()

# Test with a simple case
dsl = 'LR(C=1.0)'
print(f"Parsing: {dsl}")

try:
    result = parser.grammar.parseString(dsl, parseAll=True)
    print(f"\nResult: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result[0]: {result[0]}")
    print(f"Result[0] type: {type(result[0])}")
    
    if hasattr(result[0], '__dict__'):
        print(f"\nResult[0] attributes: {result[0].__dict__}")
    
    # Try to access tokens directly
    print(f"\nDirect access:")
    print(f"  name: {result[0].name if hasattr(result[0], 'name') else 'N/A'}")
    print(f"  pos: {result[0].pos if hasattr(result[0], 'pos') else 'N/A'}")
    print(f"  kw: {result[0].kw if hasattr(result[0], 'kw') else 'N/A'}")
    
    # Check the raw parse result before action
    print(f"\n\nTesting raw parse (before action):")
    from pyparsing import Group, Suppress, Word, alphas, alphanums, delimitedList, pyparsing_common, quotedString, Optional, Forward
    
    IDENT = Word(alphas + "_", alphanums + "_")
    EXPR = Forward()
    KWARG = Group(IDENT("key") + Suppress("=") + EXPR("value"))
    KWARGS = Optional(delimitedList(KWARG))("kw")
    CALL = Group(IDENT("name") + Suppress("(") + KWARGS + Suppress(")"))
    EXPR <<= CALL
    
    raw_result = CALL.parseString(dsl, parseAll=True)
    print(f"Raw result: {raw_result}")
    print(f"Raw result[0]: {raw_result[0]}")
    print(f"Has kw: {hasattr(raw_result[0], 'kw')}")
    if hasattr(raw_result[0], 'kw'):
        print(f"kw value: {raw_result[0].kw}")
        print(f"kw type: {type(raw_result[0].kw)}")
        if raw_result[0].kw:
            print(f"kw[0]: {raw_result[0].kw[0]}")
            print(f"kw[0] type: {type(raw_result[0].kw[0])}")
            if hasattr(raw_result[0].kw[0], 'key'):
                print(f"kw[0].key: {raw_result[0].kw[0].key}")
            if hasattr(raw_result[0].kw[0], 'value'):
                print(f"kw[0].value: {raw_result[0].kw[0].value}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

