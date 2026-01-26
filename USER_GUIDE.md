# Torque DSL - User Guide

Complete guide to using Torque Mapper and Runner for machine learning ensemble creation and evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [Config Files](#config-files)
3. [Torque Mapper](#torque-mapper)
4. [Torque Runner](#torque-runner)
5. [How They Work](#how-they-work)
6. [Quick Start Examples](#quick-start-examples)

---

## Overview

The Torque system consists of two main components:

### 1. **Torque Mapper** (`Torque_Mapper.py`)
- **Purpose**: Converts Torque DSL commands to Python code
- **Input**: Torque DSL string (e.g., `'vote(LR(C=1.0), SVM())'`)
- **Output**: JSON file containing AST and Python code
- **Does NOT**: Execute code or calculate metrics

### 2. **Torque Runner** (`runner.py`)
- **Purpose**: Executes Torque DSL commands and evaluates models
- **Input**: Config file with Torque command, data path, and settings
- **Output**: JSON file with evaluation metrics
- **Does**: Mapping + Execution + Evaluation (all in one)

### Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Option 1: Use Mapper Only (Mapping)                   │
│  Torque DSL → AST → Python Code → JSON                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Option 2: Use Runner (Complete Workflow)              │
│  Torque DSL → AST → Python Code → Execute → Metrics   │
└─────────────────────────────────────────────────────────┘
```

---

## Config Files

Config files are JSON files that contain all settings needed to run Torque commands. They make it easy to:
- Reproduce experiments
- Share configurations
- Version control settings
- Run multiple experiments

### Config File Structure

#### For Torque Mapper (`mapper_config.json`)

```json
{
  "torque_command": "vote(LR(C=1.0), SVM(kernel=\"rbf\"); voting=\"hard\")",
  "output": {
    "mapped_json_file": "torque_mapped.json"
  },
  "variable_name": "estimator",
  "print_code": false
}
```

**Fields:**
- `torque_command` (required): The Torque DSL command to convert
- `output.mapped_json_file` (optional): Where to save the mapping JSON (auto-generated if not specified)
- `variable_name` (optional): Variable name in generated Python code (default: "estimator")
- `print_code` (optional): Whether to print Python code to console (default: false)

#### For Torque Runner (`runner_config.json`)

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
    "mapped_json_file": "torque_mapped.json",
    "metrics_json_file": "torque_results.json"
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
  - `mapped_json_file`: Where to save mapping JSON
  - `metrics_json_file`: Where to save metrics JSON (default: "torque_results.json")

---

## Torque Mapper

### What It Does

Torque Mapper converts Torque DSL commands into Python code that can be executed with scikit-learn.

### Usage

```bash
python Torque_Mapper.py --config mapper_config.json
```

### What Happens

1. **Reads config file** → Gets Torque DSL command
2. **Parses DSL** → Converts to AST (Abstract Syntax Tree)
3. **Generates Python code** → Converts AST to scikit-learn Python code
4. **Saves to JSON** → Stores AST and Python code in JSON file

### Output

Creates a JSON file (e.g., `torque_mapped.json`) containing:
- `torque_command`: Original DSL command
- `ast`: Structured representation of the command
- `python_code`: Generated Python code
- `timestamp`: When it was created

### Example Output

```json
{
  "torque_command": "vote(LR(C=1.0), SVM())",
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

---

## Torque Runner

### What It Does

Torque Runner executes Torque DSL commands, runs models on data, and calculates comprehensive ML metrics.

### Usage

```bash
python runner.py --config runner_config.json
```

### What Happens

1. **Reads config file** → Gets Torque command, data path, and all settings
2. **Calls Torque Mapper internally** → Converts DSL to AST and Python code
3. **Loads data** → Reads CSV file and splits features/target
4. **Executes Python code** → Creates sklearn estimator from generated code
5. **Trains model** → Fits estimator on training data
6. **Makes predictions** → Predicts on test data
7. **Calculates metrics** → Computes accuracy, precision, recall, F1, ROC-AUC, etc.
8. **Runs cross-validation** → Performs k-fold cross-validation
9. **Saves results** → Outputs all metrics to JSON file

### Output

Creates a JSON file (e.g., `torque_results.json`) containing:
- `timestamp`: When it was run
- `source`: Information about the input (Torque command or JSON file)
- `data_info`: Dataset information (samples, features, classes)
- `metrics`: All evaluation metrics
  - `accuracy`
  - `precision_macro`, `precision_micro`, `precision_weighted`
  - `recall_macro`, `recall_micro`, `recall_weighted`
  - `f1_macro`, `f1_micro`, `f1_weighted`
  - `roc_auc` (if applicable)
  - `cohen_kappa`, `matthews_corrcoef`
  - And more...
- `cross_validation`: CV results with mean, std, min, max for each metric
- `confusion_matrix`: Confusion matrix
- `classification_report`: Detailed per-class metrics
- `status`: "success" or "error"

### When to Use Runner

- You want to evaluate a model on your data
- You need comprehensive ML metrics
- You want the complete workflow (mapping + execution + evaluation)
- You're running experiments and need reproducible results

---

## How They Work

### How Torque Mapper Works

```
┌─────────────────┐
│  Torque DSL     │
│  String         │
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
│ Torque_Mapper   │  ← Custom code generation
│ (Code Generator)│     traverses AST and builds
└────────┬────────┘     Python code strings
         │
         ▼
┌─────────────────┐
│   Python Code   │  ← Ready to execute
│     String      │
└─────────────────┘
```

**Step-by-Step:**

1. **DSL Parsing** (`dsl_mapper.py`)
   - Uses `pyparsing` library to parse the Torque DSL string
   - Recognizes patterns like `vote(...)`, `LR(C=1.0)`, etc.
   - Converts to AST dictionary structure

2. **AST Structure**
   - Tree-like dictionary representing the command structure
   - Each node has `type` ("call" or "literal")
   - Calls have `name`, `pos` (positional args), `kw` (keyword args)

3. **Code Generation** (`Torque_Mapper.py`)
   - Recursively traverses AST
   - For each model/ensemble, generates corresponding sklearn code
   - Handles imports, variable names, parameter formatting
   - Returns complete Python code string

4. **JSON Export**
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

1. **Config Loading**
   - Reads JSON config file
   - Extracts Torque command, data path, and all settings

2. **Data Loading**
   - Reads CSV file using pandas
   - Splits into features (X) and target (y)
   - Validates data format

3. **Mapping** (calls Torque Mapper internally)
   - Converts DSL to AST
   - Generates Python code
   - Saves mapping JSON (optional)

4. **Code Execution**
   - Executes generated Python code
   - Creates sklearn estimator object
   - Ready to use for training

5. **Model Training**
   - Splits data into train/test sets
   - Fits model on training data
   - Makes predictions on test data

6. **Evaluation**
   - Calculates comprehensive metrics:
     - Accuracy, Precision, Recall, F1-score
     - ROC-AUC (if applicable)
     - Cohen's Kappa, Matthews Correlation
     - Confusion Matrix
     - Classification Report
   - Runs k-fold cross-validation
   - Computes statistics (mean, std, min, max)

7. **Results Export**
   - Saves all results to JSON file
   - Includes metrics, CV results, confusion matrix, etc.

---

## Quick Start Examples

### Example 1: Mapping Only

**Step 1:** Create `mapper_config.json`:
```json
{
  "torque_command": "vote(LR(C=1.0), SVM())",
  "output": {
    "mapped_json_file": "my_mapped.json"
  }
}
```

**Step 2:** Run mapper:
```bash
python Torque_Mapper.py --config mapper_config.json
```

**Result:** Creates `my_mapped.json` with AST and Python code.

---

### Example 2: Full Execution

**Step 1:** Create `runner_config.json`:
```json
{
  "torque_command": "vote(LR(C=1.0), SVM(kernel=\"rbf\"); voting=\"hard\")",
  "data": {
    "file": "data/processed.cleveland.csv",
    "target_column": "class"
  },
  "output": {
    "metrics_json_file": "my_results.json"
  }
}
```

**Step 2:** Run runner:
```bash
python runner.py --config runner_config.json
```

**Result:** 
- Creates mapping JSON (if specified)
- Trains model on data
- Calculates metrics
- Saves results to `my_results.json`

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

---

## Summary

- **Torque Mapper**: Converts DSL → Python code (mapping only)
- **Torque Runner**: Converts DSL → Python code → Executes → Evaluates (complete workflow)
- **Config Files**: JSON files containing all settings
- **Workflow**: DSL → AST → Python Code → Execution → Metrics

Both tools use config files for easy, reproducible experiments!
