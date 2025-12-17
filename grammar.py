"""
Grammar definition for DSL generation (for grammatical evolution).

Simple grammar focused on classical ML models for ensemble learning.
"""

# Simple grammar for classical ML ensemble learning
GRAMMAR = {
    "program": ["ensemble"],
    
    # Top-level ensemble: multiple ensemble types with optional evaluation params
    "ensemble": [
        # VotingClassifier - combines multiple models
        'vote(models; ensemble_params)',
        'vote(models)',
        # StackingClassifier - stacks multiple models with a final estimator
        'stack(models; ensemble_params)',
        'stack(models)',
        # BaggingClassifier - single base model with bagging
        'bag(model; ensemble_params)',
        'bag(model)',
        # AdaBoostClassifier - single base model with boosting
        'ada(model; ensemble_params)',
        'ada(model)',
    ],
    
    # One or more base models
    "models": [
        "model",
        'model, models',
    ],
    
    # Classical ML models with model-specific parameters
    "model": [
        'LR(lr_params)',
        'SVM(svm_params)',
        'RF(rf_params)',
        'DT(dt_params)',
        'NB(nb_params)',
    ],
    
    # LR-specific parameters (LogisticRegression in sklearn)
    # Common params: C (>0), penalty (l1/l2/elasticnet), max_iter (>=1), solver, random_state
    # Note: penalty='l1' requires solver='liblinear' or 'saga'
    # Note: penalty='l2' or 'elasticnet' work with lbfgs, newton-cg, sag, saga
    # For simplicity, we'll use penalty='l2' by default (works with default solver='lbfgs')
    "lr_params": [
        'C=positive_number',
        'C=positive_number, penalty=penalty_string',
        'penalty=penalty_string',
        'C=positive_number, max_iter=int_number',
        'C=positive_number, penalty=penalty_string, max_iter=int_number',
    ],
    
    # SVM-specific parameters (SVC in sklearn)
    # Common params: C (>0), kernel (rbf/linear/poly/sigmoid), gamma (scale/auto/positive_number), probability, random_state
    # Note: kernel='precomputed' is excluded (requires special handling)
    "svm_params": [
        'C=positive_number',
        'kernel=kernel_string',
        'C=positive_number, kernel=kernel_string',
        'C=positive_number, kernel=kernel_string, gamma=gamma_value',
    ],
    
    # RF-specific parameters (RandomForestClassifier in sklearn)
    # Common params: n_estimators (>=1 int), max_depth (>=1 int or None), criterion, min_samples_split (>=2 int or (0,1] float), min_samples_leaf (>=1 int or (0,0.5] float)
    # Note: min_samples_split default=2 (must be >=2 for int), min_samples_leaf default=1 (must be >=1 for int)
    "rf_params": [
        'n_estimators=int_number',
        'max_depth=int_number',
        'criterion=criterion_string',
        'n_estimators=int_number, max_depth=int_number',
        'n_estimators=int_number, criterion=criterion_string',
        'max_depth=int_number, criterion=criterion_string',
        'n_estimators=int_number, max_depth=int_number, criterion=criterion_string',
        'n_estimators=int_number, max_depth=int_number, min_samples_split=min_samples_value',
        'n_estimators=int_number, min_samples_split=min_samples_value',
        'n_estimators=int_number, min_samples_leaf=min_samples_leaf_value',
    ],
    
    # DT-specific parameters (DecisionTreeClassifier in sklearn)
    # Common params: max_depth (>=1 int or None), criterion, min_samples_split (>=2 int or (0,1] float), min_samples_leaf (>=1 int or (0,0.5] float)
    # Note: min_samples_split default=2 (must be >=2 for int), min_samples_leaf default=1 (must be >=1 for int)
    "dt_params": [
        'max_depth=int_number',
        'criterion=criterion_string',
        'max_depth=int_number, criterion=criterion_string',
        'max_depth=int_number, min_samples_split=min_samples_value',
        'max_depth=int_number, criterion=criterion_string, min_samples_split=min_samples_value',
        'max_depth=int_number, min_samples_leaf=min_samples_leaf_value',
    ],
    
    # NB-specific parameters (GaussianNB in sklearn)
    # Common params: var_smoothing (usually default is fine)
    "nb_params": [
        "",
        'var_smoothing=small_number',
    ],
    
    # Parameter value types (matching sklearn constraints)
    "penalty_string": [
        '"l1"', '"l2"', '"elasticnet"',  # LogisticRegression penalty options (l1 needs liblinear/saga, l2/elasticnet work with multiple solvers)
    ],
    
    "kernel_string": [
        '"rbf"', '"linear"', '"poly"', '"sigmoid"',  # SVC kernel options (excludes 'precomputed' which needs special handling)
    ],
    
    "criterion_string": [
        '"gini"', '"entropy"', '"log_loss"',  # RF/DT criterion options (log_loss in sklearn >= 1.3)
    ],
    
    "gamma_value": [
        '"scale"', '"auto"',  # SVC gamma string options
        'positive_number',  # SVC gamma numeric options (must be > 0)
    ],
    
    "min_samples_value": [
        'int_number',  # Integer >= 2 for min_samples_split, >= 1 for min_samples_leaf
        'min_samples_ratio',  # Float in (0, 1] for min_samples_split or (0, 0.5] for min_samples_leaf
    ],
    
    "min_samples_ratio": [
        "0.1", "0.2", "0.3", "0.4", "0.5",  # Float ratios for min_samples_split/leaf
    ],
    
    "positive_number": [
        "0.1", "0.5", "1.0", "10", "50", "100", "200",  # Positive numbers for C, gamma, etc. (> 0)
    ],
    
    "small_number": [
        "1e-9", "1e-8", "1e-7", "1e-6",  # For var_smoothing in GaussianNB (>= 0)
    ],
    
    # Ensemble-level parameters: different params for different ensemble types
    "ensemble_params": [
        # For voting ensembles
        'voting_param',
        'voting_param, split_param',
        'voting_param, cv_param',
        'voting_param, split_param, cv_param',
        'voting_param, split_param, cv_param, scoring_param',
        # For stacking ensembles (final_estimator)
        'final_estimator_param',
        'final_estimator_param, split_param',
        'final_estimator_param, cv_param',
        'final_estimator_param, split_param, cv_param',
        'final_estimator_param, split_param, cv_param, scoring_param',
        # For bagging/ada ensembles (n_estimators)
        'n_estimators_param',
        'n_estimators_param, split_param',
        'n_estimators_param, cv_param',
        'n_estimators_param, split_param, cv_param',
        'n_estimators_param, split_param, cv_param, scoring_param',
        # Generic (no ensemble-specific params)
        'split_param',
        'cv_param',
        'scoring_param',
        'split_param, cv_param',
        'split_param, cv_param, scoring_param',
    ],
    
    # VotingClassifier specific parameter
    # Note: voting must be "hard" or "soft" (not arbitrary strings)
    "voting_param": [
        'voting=voting_string',
    ],
    
    "voting_string": [
        '"hard"', '"soft"',  # VotingClassifier voting options
    ],
    
    # StackingClassifier specific parameter (final estimator)
    "final_estimator_param": [
        'final_estimator=model_name',
    ],
    
    # Bagging/AdaBoost specific parameter (number of estimators)
    "n_estimators_param": [
        'n_estimators=int_number',
    ],
    
    # Model names for final_estimator in stacking
    "model_name": [
        '"LR"', '"SVM"', '"RF"', '"DT"', '"NB"',
    ],
    
    # Train / test split parameters (e.g. test_size=0.2 -> 20/80 split)
    "split_param": [
        'test_size=split_ratio',
        'train_size=split_ratio',
    ],
    
    # Cross‑validation parameters
    "cv_param": [
        'cv_folds=int_number',
    ],
    
    # Scoring metric for evaluation
    "scoring_param": [
        'scoring=score_metric',
    ],
    
    # Generic numeric values for hyperparameters
    "number": [
        "0.1", "0.5", "1.0", "10", "50", "100", "200",
    ],
    
    # Split ratios for train/test
    "split_ratio": [
        "0.1", "0.2", "0.25", "0.3", "0.4", "0.5",
    ],
    
    # Integer values (for folds, n_estimators, max_depth, max_iter, min_samples_split, etc.)
    # These must be integers >= 1 for sklearn parameters
    "int_number": [
        "1", "2", "3", "5", "10", "50", "100", "200",
    ],
    
    # String values (kernels, criteria, voting modes, etc.)
    "string": [
        '"rbf"', '"linear"', '"poly"', '"hard"', '"soft"', '"gini"', '"entropy"',
    ],
    
    # Scoring metrics for evaluation
    "score_metric": [
        '"accuracy"', '"f1"', '"precision"', '"recall"',
    ],
}


def generate_dsl_from_grammar(rule="program", max_depth=10, current_depth=0):
    """
    Generate a DSL string from the grammar (for grammatical evolution).
    
    Args:
        rule: Starting grammar rule
        max_depth: Maximum recursion depth
        current_depth: Current depth
        
    Returns:
        Generated DSL string
    """
    # Base case: if we've exceeded max depth, try to return a terminal
    if current_depth >= max_depth:
        if rule in ["number", "string", "split_ratio", "int_number", "score_metric", "model_name",
                    "penalty_string", "kernel_string", "criterion_string", "gamma_value", "small_number",
                    "positive_number", "min_samples_value", "min_samples_ratio", "min_samples_leaf_value",
                    "min_samples_split_int", "min_samples_leaf_ratio", "voting_string"]:
            if rule in GRAMMAR:
                import random
                choice = random.choice(GRAMMAR[rule])
                # Remove quotes from string-like terminals when needed
                if choice.startswith('"') and choice.endswith('"'):
                    return choice[1:-1]
                return choice
        return ""
    
    # If rule is not in grammar, it's a terminal - return as is
    if rule not in GRAMMAR:
        return rule
    
    # Choose a random production
    import random
    production = random.choice(GRAMMAR[rule])
    
    # Parse the production - handle it as a template
    result = production
    
    # Find all non-terminals in the production and expand them
    import re
    # Find words that are non-terminals (in GRAMMAR)
    # Match word boundaries to avoid partial matches
    pattern = r'\b(' + '|'.join(re.escape(k) for k in GRAMMAR.keys()) + r')\b'
    
    def replace_non_terminal(match):
        non_terminal = match.group(1)
        if current_depth < max_depth:
            expanded = generate_dsl_from_grammar(non_terminal, max_depth, current_depth + 1)
            return expanded if expanded else non_terminal
        return non_terminal
    
    # Replace all non-terminals
    result = re.sub(pattern, replace_non_terminal, result)
    
    # Handle empty params - if result is empty or just whitespace, return empty
    if result.strip() == "":
        return ""
    
    # Handle special cases: assignments and quoted strings
    # For assignments like "C=number", we need to expand "number"
    # This needs to be recursive to handle nested non-terminals
    def expand_assignments(text, depth=0, max_assign_depth=5):
        if depth >= max_assign_depth:
            return text  # Prevent infinite recursion
        # Find patterns like "key=non_terminal"
        pattern = r'(\w+)=(\w+)'
        def replace_assignment(match):
            key = match.group(1)
            value_rule = match.group(2)
            if value_rule in GRAMMAR and current_depth + depth < max_depth:
                expanded_value = generate_dsl_from_grammar(value_rule, max_depth, current_depth + depth + 1)
                # If expansion still contains non-terminals, recurse
                if expanded_value and any(nt in GRAMMAR for nt in expanded_value.split() if nt in GRAMMAR):
                    expanded_value = expand_assignments(expanded_value, depth + 1, max_assign_depth)
                # Handle empty expansion
                if expanded_value == "":
                    return ""
                # Add quotes for string types
                if value_rule in ["string", "penalty_string", "kernel_string", "criterion_string", "score_metric", "voting_string"]:
                    if not expanded_value.startswith('"'):
                        return f'{key}="{expanded_value}"'
                return f'{key}={expanded_value}'
            return match.group(0)
        new_text = re.sub(pattern, replace_assignment, text)
        # If we made changes and there are still non-terminals, recurse
        if new_text != text and any(nt in GRAMMAR for nt in new_text.split() if nt in GRAMMAR):
            return expand_assignments(new_text, depth + 1, max_assign_depth)
        return new_text
    
    result = expand_assignments(result)
    
    # Clean up empty params: remove empty parentheses content
    # Handle cases like "LR()" where params expanded to empty
    result = re.sub(r'\(\s*\)', '()', result)  # Normalize empty ()
    result = re.sub(r'\(\s*,\s*\)', '()', result)  # Remove empty with comma
    result = re.sub(r'\(\s*,\s*', '(', result)  # Remove leading comma
    result = re.sub(r',\s*\)', ')', result)  # Remove trailing comma
    result = re.sub(r',\s*,', ',', result)  # Remove double commas
    
    # Remove any remaining quoted non-terminals (like '"rbf"' should become "rbf")
    result = re.sub(r'"(\w+)"', r'"\1"', result)
    
    return result
