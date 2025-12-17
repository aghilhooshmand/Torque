"""
Test script to verify the DSL parser and compiler work correctly.
"""

from dsl_parser import parse_dsl_to_ast
from compiler import compile_ast_to_estimator, dsl_to_sklearn_estimator
from executor import execute_dsl
from evaluator import evaluate_program
import numpy as np
from sklearn.datasets import make_classification

# Create test data
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Test DSL string
dsl = 'vote(LR(C=1.0), SVM(C=1.0, kernel="rbf"); voting="hard")'

print("Testing DSL parser and compiler...")
print(f"DSL: {dsl}")
print()

try:
    # Test parsing
    print("1. Parsing DSL...")
    ast = parse_dsl_to_ast(dsl)
    print(f"   AST type: {ast.get('type')}")
    print(f"   AST name: {ast.get('name')}")
    print()
    
    # Test compilation
    print("2. Compiling AST to sklearn estimator...")
    model = compile_ast_to_estimator(ast)
    print(f"   Model type: {type(model)}")
    print(f"   Model: {model}")
    print()
    
    # Test execution (train/test split)
    print("3. Executing DSL (train/test split)...")
    accuracy, fitted_model = execute_dsl(dsl, X, y, return_model=True)
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Fitted model type: {type(fitted_model)}")
    print()
    
    # Test evaluation (cross-validation)
    print("4. Evaluating DSL (cross-validation)...")
    cv_score = evaluate_program(dsl, X, y, scoring="accuracy", folds=5)
    print(f"   CV Score: {cv_score:.4f}")
    print()
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

