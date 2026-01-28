# Torque DSL - User Guide

Complete guide to using Torque Mapper and Runner for machine learning ensemble creation and evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [File Naming Convention](#file-naming-convention)
3. [Config Files](#config-files)
4. [Torque Mapper](#torque-mapper)
5. [Torque Runner](#torque-runner)
6. [How They Work](#how-they-work)
7. [Quick Start Examples](#quick-start-examples)

---

## Overview

The Torque system consists of two main components that work together to convert Torque DSL commands into executable machine learning models and evaluate them:

### 1. **Torque Mapper** (`Torque_mapper.py`)
- **Purpose**: Converts Torque DSL commands to Python code
- **Input**: Torque DSL string (e.g., `'vote(LR(C=1.0), SVM())'`)
- **Output**: JSON file containing AST and Python code (`Torque_mapper_result.json`)
- **Does NOT**: Execute code or calculate metrics
- **Key Feature**: Pure mapping functionality - no execution

### 2. **Torque Runner** (`Torque_runner.py`)
- **Purpose**: Executes Torque DSL commands and evaluates models
- **Input**: Config file with Torque command, data path, and all settings
- **Output**: JSON file with comprehensive evaluation metrics (`Torque_runner_result.json`)
- **Does**: Complete workflow - Mapping + Execution + Evaluation (all in one)
- **Key Feature**: Calls Torque Mapper internally, then executes and evaluates

### Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Option 1: Use Mapper Only (Mapping)                   │
│  Torque DSL → AST → Python Code → JSON                │
│  Output: Torque_mapper_result.json                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Option 2: Use Runner (Complete Workflow)              │
│  Config → Load Data → DSL → AST → Code → Execute → Metrics │
│  Output: Torque_runner_result.json                     │
└─────────────────────────────────────────────────────────┘
```

---

## File Naming Convention

All files follow a consistent naming pattern with the `Torque_` prefix:

### Python Files
- `Torque_mapper.py` - Mapping module (DSL → AST → Python code)
- `Torque_runner.py` - Execution module (Complete workflow)

### Config Files
- `Torque_mapper_config.json` - Configuration for mapper
- `Torque_runner_config.json` - Configuration for runner

### Output Files
- `Torque_mapper_result.json` - Output from mapper (AST + Python code)
- `Torque_runner_result.json` - Output from runner (Metrics + Results)

---

## Config Files

**All settings are now read from JSON config files** - no command-line arguments needed (except `--config` flag).

Config files make it easy to:
- Reproduce experiments
- Share configurations
- Version control settings
- Run multiple experiments with different configs

### Config File Structure

#### For Torque Mapper (`Torque_mapper_config.json`)

```json
{
  "torque_command": "vote(LR(C=1.0), SVM(kernel=\"rbf\"); voting=\"hard\")",
  "output": {
    "mapped_json_file": "Torque_mapper_result.json"
  },
  "variable_name": "estimator",
  "print_code": false
}
```

**Fields:**
- `torque_command` (required): The Torque DSL command to convert
- `output.mapped_json_file` (optional): Where to save the mapping JSON (default: `Torque_mapper_result.json`)
- `variable_name` (optional): Variable name in generated Python code (default: "estimator")
- `print_code` (optional): Whether to print Python code to console (default: false)

#### For Torque Runner (`Torque_runner_config.json`)

```json
{
  "torque_command": "vote(LR(C=1.0), SVM(kernel=\"rbf\"); voting=\"hard\")",
  "data": {
    "file": "data/processed.cleveland.csv",
    "target_column": "class",
    "delimiter": ",",
    "header": true
  },
  "splitting": {
    "test_size": 0.3,
    "random_state": 42
  },
  "evaluation": {
    "cv_folds": 5,
    "random_state": 42
  },
  "output": {
    "mapped_json_file": "Torque_mapper_result.json",
    "metrics_json_file": "Torque_runner_result.json"
  }
}
```

**Fields:**
- `torque_command` (required): The Torque DSL command to execute
- `data` (required): Data file settings
  - `file`: Path to CSV data file
  - `target_column`: Name of target column (uses last column if not specified)
  - `delimiter`: CSV delimiter (default: ",")
  - `header`: Whether CSV has header row (default: true)
- `splitting` (optional): Train/test split settings
  - `test_size`: Proportion for test set (default: 0.3)
  - `random_state`: Random seed (default: 42)
- `evaluation` (optional): Cross-validation settings
  - `cv_folds`: Number of CV folds (default: 5)
  - `random_state`: Random seed (default: 42)
- `output` (optional): Output file paths
  - `mapped_json_file`: Where to save mapping JSON (optional, auto-generated if not specified)
  - `metrics_json_file`: Where to save metrics JSON (default: `Torque_runner_result.json`)

---

## Torque Mapper

### What It Does

`Torque_mapper.py` converts Torque DSL commands into Python code that can be executed with scikit-learn. It performs **mapping only** - no code execution.

### Key Methods

The `TorqueMapper` class provides these methods:

1. **`dsl_to_ast(torque_command: str) -> Dict`**
   - Converts Torque DSL string to AST (Abstract Syntax Tree)
   - Returns dictionary representation of the command structure

2. **`ast_to_python(ast: Dict, variable_name: str) -> str`**
   - Converts AST to runnable Python code string
   - Includes all necessary sklearn imports
   - Returns complete Python code ready for execution

3. **`map_to_python(torque_command: str, variable_name: str) -> str`**
   - Convenience method: DSL → Python code in one step
   - Combines `dsl_to_ast()` and `ast_to_python()`

4. **`export_to_json(torque_command: str, output_file: str, variable_name: str) -> Dict`**
   - Main method: Converts DSL to AST and Python code, saves to JSON
   - Returns dictionary with all mapping information

### Usage

**Command-line (with config file):**
```bash
python Torque_mapper.py --config Torque_mapper_config.json
```

**Python API:**
```python
from Torque_mapper import TorqueMapper

mapper = TorqueMapper()

# Option 1: Export to JSON (recommended)
result = mapper.export_to_json(
    torque_command='vote(LR(C=1.0), SVM())',
    output_file='Torque_mapper_result.json'
)

# Option 2: Get Python code directly
python_code = mapper.map_to_python('vote(LR(C=1.0), SVM())')
print(python_code)

# Option 3: Step-by-step
ast = mapper.dsl_to_ast('vote(LR(C=1.0), SVM())')
python_code = mapper.ast_to_python(ast)
```

### What Happens

1. **Reads config file** → Gets Torque DSL command and settings
2. **Parses DSL** → Uses `dsl_mapper.py` (pyparsing) to convert to AST
3. **Generates Python code** → Recursively traverses AST and builds sklearn code
4. **Saves to JSON** → Stores AST, Python code, and metadata in JSON file

### Output

Creates a JSON file (e.g., `Torque_mapper_result.json`) containing:
- `torque_command`: Original DSL command
- `timestamp`: When it was created
- `variable_name`: Variable name used in generated code
- `ast`: Structured dictionary representation of the command
- `python_code`: Complete Python code string with imports

### Example Output

```json
{
  "torque_command": "vote(LR(C=1.0), SVM())",
  "timestamp": "2025-01-26T10:30:00",
  "variable_name": "estimator",
  "ast": {
    "type": "call",
    "name": "vote",
    "pos": [
      {
        "type": "call",
        "name": "LR",
        "kw": {"C": {"type": "literal", "value": 1.0}}
      },
      {
        "type": "call",
        "name": "SVM",
        "kw": {}
      }
    ],
    "kw": {}
  },
  "python_code": "from sklearn.ensemble import VotingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\n\nbase_estimator_0 = LogisticRegression(C=1.0)\nbase_estimator_1 = SVC()\nestimator = VotingClassifier(estimators=[('model_0', base_estimator_0), ('model_1', base_estimator_1)])"
}
```

### When to Use Mapper

- You only need the Python code (for inspection, documentation, or manual execution)
- You want to review the generated code before running
- You're developing or debugging the mapping process
- You want to save the mapping for later use
- You need to generate code without executing it

---

## Torque Runner

### What It Does

`Torque_runner.py` executes Torque DSL commands, runs models on data, and calculates comprehensive ML metrics. It performs the **complete workflow** - mapping, execution, and evaluation.

### Key Methods

The `TorqueRunner` class provides these methods:

1. **`run_dsl(torque_command: str, X: np.ndarray, y: np.ndarray, ...) -> Dict`**
   - Main method: Accepts Torque DSL command directly
   - Calls `TorqueMapper.export_to_json()` internally to create mapping
   - Executes Python code and evaluates model
   - Returns comprehensive metrics dictionary

2. **`run_from_json(json_file: str, X: np.ndarray, y: np.ndarray, ...) -> Dict`**
   - Runs from existing mapping JSON file (from `Torque_mapper.py`)
   - Executes Python code and evaluates model
   - Useful when you already have a mapping JSON

3. **`run_ast(ast: Dict, X: np.ndarray, y: np.ndarray, ...) -> Dict`**
   - Runs directly from AST dictionary
   - Uses `compiler.py` to create sklearn estimator
   - Bypasses code generation step

4. **`load_config(config_file: str) -> Dict`** (static method)
   - Loads JSON config file
   - Returns configuration dictionary

5. **`load_data_from_config(config: Dict) -> Tuple`** (static method)
   - Loads CSV data file based on config settings
   - Returns (X, y) tuple (features and target)

### Usage

**Command-line (with config file - recommended):**
```bash
python Torque_runner.py --config Torque_runner_config.json
```

**Python API:**
```python
from Torque_runner import TorqueRunner
import numpy as np

runner = TorqueRunner()

# Option 1: Run from DSL command (recommended)
results = runner.run_dsl(
    torque_command='vote(LR(C=1.0), SVM())',
    X=X, y=y,
    metrics_json_file='Torque_runner_result.json'
)

# Option 2: Run from existing mapping JSON
results = runner.run_from_json(
    json_file='Torque_mapper_result.json',
    X=X, y=y,
    output_file='Torque_runner_result.json'
)

# Option 3: Load config and run
config = TorqueRunner.load_config('Torque_runner_config.json')
X, y = TorqueRunner.load_data_from_config(config)
results = runner.run_dsl(
    torque_command=config['torque_command'],
    X=X, y=y
)
```

### What Happens

1. **Reads config file** → Gets Torque command, data path, and all settings
2. **Loads data** → Reads CSV file using pandas, splits into features (X) and target (y)
3. **Calls Torque Mapper internally** → Converts DSL to AST and Python code
4. **Saves mapping JSON** → Optionally saves mapping result (if specified in config)
5. **Executes Python code** → Runs generated code to create sklearn estimator
6. **Splits data** → Train/test split (default: 70/30)
7. **Trains model** → Fits estimator on training data
8. **Makes predictions** → Predicts on test data (and probabilities if available)
9. **Calculates metrics** → Computes comprehensive ML metrics
10. **Runs cross-validation** → Performs k-fold cross-validation
11. **Generates reports** → Creates confusion matrix and classification report
12. **Saves results** → Outputs all results to JSON file

### Output

Creates a JSON file (e.g., `Torque_runner_result.json`) containing:
- `timestamp`: When it was run
- `source`: Information about input (Torque command or JSON file)
- `data_info`: Dataset information (samples, features, classes, distribution)
- `parameters`: Evaluation parameters (test_size, cv_folds, random_state)
- `metrics`: All evaluation metrics
  - `accuracy`
  - `precision_macro`, `precision_micro`, `precision_weighted`
  - `recall_macro`, `recall_micro`, `recall_weighted`
  - `f1_macro`, `f1_micro`, `f1_weighted`
  - `roc_auc` (if applicable)
  - `cohen_kappa`, `matthews_corrcoef`
  - `average_precision`, `log_loss` (if probabilities available)
- `cross_validation`: CV results with mean, std, min, max for each metric
- `confusion_matrix`: Confusion matrix with labels
- `classification_report`: Detailed per-class metrics
- `status`: "success" or "error"
- `error`: Error message (if status is "error")

### When to Use Runner

- You want to evaluate a model on your data
- You need comprehensive ML metrics
- You want the complete workflow (mapping + execution + evaluation)
- You're running experiments and need reproducible results
- You want to compare different models or configurations

---

## How They Work

### How Torque Mapper Works

```
┌─────────────────┐
│  Config File    │
│  (JSON)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract DSL   │
│  Command        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  dsl_mapper.py  │  ← Uses pyparsing library
│  (Parser)       │     to parse DSL string
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      AST        │  ← Structured dictionary
│  (Dictionary)   │     representation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Torque_mapper   │  ← Custom code generation
│ (Code Generator)│     traverses AST and builds
└────────┬────────┘     Python code strings
         │
         ▼
┌─────────────────┐
│   Python Code   │  ← Ready to execute
│     String      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   JSON File     │  ← Saves AST + Python code
│ Torque_mapper_  │
│ result.json     │
└─────────────────┘
```

**Step-by-Step:**

1. **Config Loading** (`Torque_mapper.py`)
   - Reads JSON config file
   - Extracts `torque_command` and settings

2. **DSL Parsing** (`dsl_mapper.py`)
   - Uses `pyparsing` library to parse the Torque DSL string
   - Recognizes patterns like `vote(...)`, `LR(C=1.0)`, etc.
   - Converts to AST dictionary structure

3. **AST Structure**
   - Tree-like dictionary representing the command structure
   - Each node has `type` ("call" or "literal")
   - Calls have `name`, `pos` (positional args), `kw` (keyword args)

4. **Code Generation** (`Torque_mapper.py`)
   - Recursively traverses AST using `_ast_to_python()`
   - For models: Uses `_build_model_code()` to generate sklearn model code
   - For ensembles: Uses `_build_ensemble_code()` to generate ensemble code
   - Handles imports, variable names, parameter formatting
   - Returns complete Python code string

5. **JSON Export**
   - Saves AST and Python code to JSON file
   - Includes metadata (timestamp, variable name, etc.)

### How Torque Runner Works

```
┌─────────────────┐
│  Config File    │
│  (JSON)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Config    │  ← Reads JSON config
│  Load Data      │     Loads CSV data file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Torque Mapper  │  ← Calls mapper internally
│  (DSL → Code)   │     to convert DSL to code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execute Code   │  ← Runs Python code
│  Create Model   │     Creates sklearn estimator
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train/Test     │  ← Splits data, trains model
│  Predict        │     Makes predictions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculate      │  ← Computes all metrics
│  Metrics        │     Cross-validation, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Results   │  ← Outputs to JSON
│  (JSON)         │
└─────────────────┘
```

**Step-by-Step:**

1. **Config Loading** (`Torque_runner.py`)
   - Reads JSON config file using `load_config()`
   - Extracts Torque command, data path, and all settings

2. **Data Loading** (`Torque_runner.py`)
   - Uses `load_data_from_config()` to read CSV file
   - Splits into features (X) and target (y)
   - Validates data format and column existence

3. **Mapping** (calls `Torque_mapper.py` internally)
   - Creates `TorqueMapper` instance
   - Calls `mapper.export_to_json()` to convert DSL to AST and Python code
   - Optionally saves mapping JSON (if specified in config)

4. **Code Execution** (`Torque_runner.py`)
   - Reads Python code from mapping JSON
   - Executes code using `exec()` in isolated namespace
   - Creates sklearn estimator object
   - Ready to use for training

5. **Model Training** (`Torque_runner.py`)
   - Splits data into train/test sets using `train_test_split()`
   - Fits model on training data using `estimator.fit()`
   - Makes predictions on test data using `estimator.predict()`
   - Gets probabilities if available using `estimator.predict_proba()`

6. **Evaluation** (`Torque_runner.py`)
   - Calculates comprehensive metrics using `_calculate_all_metrics()`:
     - Accuracy, Precision, Recall, F1-score (macro, micro, weighted)
     - ROC-AUC (if probabilities available)
     - Cohen's Kappa, Matthews Correlation
     - Average Precision, Log Loss (if applicable)
   - Runs k-fold cross-validation using `_run_cross_validation()`
   - Generates confusion matrix using `confusion_matrix()`
   - Creates classification report using `classification_report()`

7. **Results Export** (`Torque_runner.py`)
   - Saves all results to JSON file
   - Includes metrics, CV results, confusion matrix, classification report
   - Adds metadata (timestamp, source info, data info, parameters)

---

## Quick Start Examples

### Example 1: Mapping Only

**Step 1:** Create `Torque_mapper_config.json`:
```json
{
  "torque_command": "vote(LR(C=1.0), SVM())",
  "output": {
    "mapped_json_file": "Torque_mapper_result.json"
  }
}
```

**Step 2:** Run mapper:
```bash
python Torque_mapper.py --config Torque_mapper_config.json
```

**Result:** Creates `Torque_mapper_result.json` with AST and Python code.

---

### Example 2: Full Execution

**Step 1:** Create `Torque_runner_config.json`:
```json
{
  "torque_command": "vote(LR(C=1.0), SVM(kernel=\"rbf\"); voting=\"hard\")",
  "data": {
    "file": "data/processed.cleveland.csv",
    "target_column": "class"
  },
  "output": {
    "metrics_json_file": "Torque_runner_result.json"
  }
}
```

**Step 2:** Run runner:
```bash
python Torque_runner.py --config Torque_runner_config.json
```

**Result:** 
- Creates mapping JSON (if specified)
- Trains model on data
- Calculates metrics
- Saves results to `Torque_runner_result.json`

---

### Example 3: Different Ensemble Types

**Voting Classifier:**
```json
{
  "torque_command": "vote(LR(C=1.0), SVM(), RF(n_estimators=100); voting=\"hard\")",
  ...
}
```

**Stacking Classifier:**
```json
{
  "torque_command": "stack(LR(), RF(); final_estimator=LR())",
  ...
}
```

**Bagging Classifier:**
```json
{
  "torque_command": "bag(RF(n_estimators=50); n_estimators=10)",
  ...
}
```

**AdaBoost Classifier:**
```json
{
  "torque_command": "ada(DT(max_depth=3); n_estimators=50)",
  ...
}
```

---

## Torque DSL Syntax

### Models

- `LR(C=1.0)` - Logistic Regression
- `SVM(kernel="rbf", C=1.0)` - Support Vector Machine
- `RF(n_estimators=100)` - Random Forest
- `DT(max_depth=5)` - Decision Tree
- `NB()` - Naive Bayes

### Ensembles

- `vote(model1, model2; voting="hard")` - Voting Classifier
- `stack(model1, model2; final_estimator=LR())` - Stacking Classifier
- `bag(model; n_estimators=10)` - Bagging Classifier
- `ada(model; n_estimators=50)` - AdaBoost Classifier

### Parameters

- Use `;` to separate ensemble options from base models
- Keyword arguments: `key=value`
- String values: Use quotes `"value"`
- Numbers: No quotes `1.0`, `100`

---

## Key Changes and Features

### What Changed

1. **File Naming**: All files now use consistent `Torque_` prefix
   - `Torque_mapper.py` (was `Torque_Mapper.py`)
   - `Torque_runner.py` (was `runner.py`)
   - Config files: `Torque_mapper_config.json`, `Torque_runner_config.json`
   - Output files: `Torque_mapper_result.json`, `Torque_runner_result.json`

2. **Config-File Only**: Everything is configured via JSON files
   - No complex command-line arguments
   - All settings in one place
   - Easy to reproduce and share

3. **Clear Separation**: 
   - `Torque_mapper.py`: Mapping only (DSL → AST → Python code)
   - `Torque_runner.py`: Complete workflow (calls mapper internally)

4. **Runner Calls Mapper**: 
   - `Torque_runner.py` internally uses `Torque_mapper.py`
   - No need to run mapper separately (unless you want just the code)

### Key Features

- **Modular Design**: Mapper and Runner can be used separately or together
- **Config-Driven**: All settings in JSON files for reproducibility
- **Comprehensive Metrics**: Runner calculates all standard ML metrics
- **Flexible Usage**: Can use Python API or command-line interface
- **Clear Output**: Well-structured JSON files with all information

---

## Tips and Best Practices

1. **Use Config Files**: Always use config files for reproducibility
2. **Version Control**: Track config files in git to reproduce experiments
3. **Descriptive Names**: Use clear names for output files
4. **Check Data Format**: Ensure CSV has headers and correct target column
5. **Review Generated Code**: Use mapper first to review Python code before running
6. **Multiple Configs**: Create different configs for different experiments
7. **Default Values**: Many settings have sensible defaults, you don't need to specify everything

---

## Troubleshooting

### Error: Config file not found
- Check the file path is correct
- Ensure you're running from the project root directory

### Error: Data file not found
- Verify the path in config file is correct
- Use relative paths from project root

### Error: Target column not found
- Check column name matches exactly (case-sensitive)
- If not specified, last column is used automatically

### Error: Unknown model or ensemble
- Check Torque DSL syntax
- Ensure model/ensemble names are correct (LR, SVM, RF, DT, NB, vote, stack, bag, ada)

### Error: Module not found
- Ensure you're in the project root directory
- Check that `Torque_mapper.py` and `Torque_runner.py` are in the same directory

---

## Summary

- **Torque Mapper**: Converts DSL → AST → Python code (mapping only)
- **Torque Runner**: Converts DSL → AST → Python code → Executes → Evaluates (complete workflow)
- **Config Files**: JSON files containing all settings
- **Workflow**: Config → Data → Mapping → Execution → Metrics → JSON

Both tools use config files for easy, reproducible experiments!
