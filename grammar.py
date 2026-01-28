"""
Grammar definition for DSL generation (for grammatical evolution).

Simple grammar focused on classical ML models for ensemble learning.
BNF (Backus-Naur Form) style grammar definition.
"""

# BNF-style grammar for classical ML ensemble learning
# Compatible with BNF Playground (https://bnfplayground.pauliankline.com/)
# All terminals are quoted as required by the BNF Playground
GRAMMAR_BNF = """
<program> ::= <ensemble>

<ensemble> ::= "vote" "(" <models> ";" <ensemble_params> ")"
             | "vote" "(" <models> ")"
             | "stack" "(" <models> ";" <ensemble_params> ")"
             | "stack" "(" <models> ")"
             | "bag" "(" <model> ";" <ensemble_params> ")"
             | "bag" "(" <model> ")"
             | "ada" "(" <model> ";" <ensemble_params> ")"
             | "ada" "(" <model> ")"

<models> ::= <model>
           | <model> "," <models>

<model> ::= "LR" "(" <lr_params> ")"
          | "SVM" "(" <svm_params> ")"
          | "RF" "(" <rf_params> ")"
          | "DT" "(" <dt_params> ")"
          | "NB" "(" <nb_params> ")"

<lr_params> ::= "C" "=" <positive_number>
              | "C" "=" <positive_number> "," "penalty" "=" <penalty_string>
              | "penalty" "=" <penalty_string>
              | "C" "=" <positive_number> "," "max_iter" "=" <int_number>
              | "C" "=" <positive_number> "," "penalty" "=" <penalty_string> "," "max_iter" "=" <int_number>

<svm_params> ::= "C" "=" <positive_number>
              | "kernel" "=" <kernel_string>
              | "C" "=" <positive_number> "," "kernel" "=" <kernel_string>
              | "C" "=" <positive_number> "," "kernel" "=" <kernel_string> "," "gamma" "=" <gamma_value>

<rf_params> ::= "n_estimators" "=" <int_number>
             | "max_depth" "=" <int_number>
             | "criterion" "=" <criterion_string>
             | "n_estimators" "=" <int_number> "," "max_depth" "=" <int_number>
             | "n_estimators" "=" <int_number> "," "criterion" "=" <criterion_string>
             | "max_depth" "=" <int_number> "," "criterion" "=" <criterion_string>
             | "n_estimators" "=" <int_number> "," "max_depth" "=" <int_number> "," "criterion" "=" <criterion_string>
             | "n_estimators" "=" <int_number> "," "max_depth" "=" <int_number> "," "min_samples_split" "=" <min_samples_value>
             | "n_estimators" "=" <int_number> "," "min_samples_split" "=" <min_samples_value>
             | "n_estimators" "=" <int_number> "," "min_samples_leaf" "=" <min_samples_leaf_value>

<dt_params> ::= "max_depth" "=" <int_number>
             | "criterion" "=" <criterion_string>
             | "max_depth" "=" <int_number> "," "criterion" "=" <criterion_string>
             | "max_depth" "=" <int_number> "," "min_samples_split" "=" <min_samples_value>
             | "max_depth" "=" <int_number> "," "criterion" "=" <criterion_string> "," "min_samples_split" "=" <min_samples_value>
             | "max_depth" "=" <int_number> "," "min_samples_leaf" "=" <min_samples_leaf_value>

<nb_params> ::= "var_smoothing" "=" <small_number>

<penalty_string> ::= "l1" | "l2" | "elasticnet"

<kernel_string> ::= "rbf" | "linear" | "poly" | "sigmoid"

<criterion_string> ::= "gini" | "entropy" | "log_loss"

<gamma_value> ::= "scale" | "auto" | <positive_number>

<min_samples_value> ::= <int_number> | <min_samples_ratio>

<min_samples_ratio> ::= "0.1" | "0.2" | "0.3" | "0.4" | "0.5"

<min_samples_leaf_value> ::= <int_number> | <min_samples_ratio>

<positive_number> ::= "0.1" | "0.5" | "1.0" | "10" | "50" | "100" | "200"

<small_number> ::= "1e-9" | "1e-8" | "1e-7" | "1e-6"

<ensemble_params> ::= <voting_param>
                    | <voting_param> "," <split_param>
                    | <voting_param> "," <cv_param>
                    | <voting_param> "," <split_param> "," <cv_param>
                    | <voting_param> "," <split_param> "," <cv_param> "," <scoring_param>
                    | <final_estimator_param>
                    | <final_estimator_param> "," <split_param>
                    | <final_estimator_param> "," <cv_param>
                    | <final_estimator_param> "," <split_param> "," <cv_param>
                    | <final_estimator_param> "," <split_param> "," <cv_param> "," <scoring_param>
                    | <n_estimators_param>
                    | <n_estimators_param> "," <split_param>
                    | <n_estimators_param> "," <cv_param>
                    | <n_estimators_param> "," <split_param> "," <cv_param>
                    | <n_estimators_param> "," <split_param> "," <cv_param> "," <scoring_param>
                    | <split_param>
                    | <cv_param>
                    | <scoring_param>
                    | <split_param> "," <cv_param>
                    | <split_param> "," <cv_param> "," <scoring_param>

<voting_param> ::= "voting" "=" <voting_string>

<voting_string> ::= "hard" | "soft"

<final_estimator_param> ::= "final_estimator" "=" <model_name>

<n_estimators_param> ::= "n_estimators" "=" <int_number>

<model_name> ::= "LR" | "SVM" | "RF" | "DT" | "NB"

<split_param> ::= "test_size" "=" <split_ratio>
               | "train_size" "=" <split_ratio>

<cv_param> ::= "cv_folds" "=" <int_number>

<scoring_param> ::= "scoring" "=" <score_metric>

<split_ratio> ::= "0.1" | "0.2" | "0.25" | "0.3" | "0.4" | "0.5"

<int_number> ::= "1" | "2" | "3" | "5" | "10" | "50" | "100" | "200"

<score_metric> ::= "accuracy" | "f1" | "precision" | "recall"
"""


def _parse_bnf_grammar(bnf_text):
    """
    Parse BNF grammar text into a dictionary format compatible with the existing generator.
    
    Args:
        bnf_text: BNF grammar as string
        
    Returns:
        Dictionary mapping non-terminal names to lists of productions
    """
    grammar = {}
    lines = bnf_text.strip().split('\n')
    
    current_rule = None
    current_productions = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Check if line starts a new rule: <rule_name> ::=
        if '::=' in line:
            # Save previous rule if exists
            if current_rule:
                grammar[current_rule] = current_productions if current_productions else [""]
            
            # Extract rule name
            parts = line.split('::=')
            rule_part = parts[0].strip()
            # Remove < > from rule name
            rule_name = rule_part.strip('<>').strip()
            current_rule = rule_name
            current_productions = []
            
            # Check if there's a production on the same line
            if len(parts) > 1:
                prod = parts[1].strip()
                if prod:
                    current_productions.append(prod)
                else:
                    # Empty production
                    current_productions.append("")
        
        # Check if line continues with | (alternative production)
        elif line.startswith('|'):
            prod = line[1:].strip()
            # Allow empty productions (for nb_params)
            current_productions.append(prod)
        
        # Otherwise, it's a continuation of the current production
        elif current_rule:
            current_productions[-1] += ' ' + line
    
    # Save last rule
    if current_rule:
        grammar[current_rule] = current_productions if current_productions else [""]
    
    # Split productions that contain | into separate alternatives
    # This handles cases like: <rule> ::= "a" | "b" | "c"
    final_grammar = {}
    for rule_name, productions in grammar.items():
        final_productions = []
        for prod in productions:
            # Split on | but preserve quoted strings
            # Simple approach: split on | that's not inside quotes
            parts = []
            current_part = ""
            in_quotes = False
            for char in prod:
                if char == '"':
                    in_quotes = not in_quotes
                    current_part += char
                elif char == '|' and not in_quotes:
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            if current_part.strip():
                parts.append(current_part.strip())
            
            if parts:
                final_productions.extend(parts)
            else:
                final_productions.append(prod)
        
        final_grammar[rule_name] = final_productions if final_productions else [""]
    
    # Strip quotes from terminals in productions for internal use
    # This converts "vote" "(" <models> to vote(<models> for DSL generation
    import re
    cleaned_grammar = {}
    for rule_name, productions in final_grammar.items():
        cleaned_productions = []
        for prod in productions:
            # Remove quotes from terminals: "vote" -> vote
            cleaned = re.sub(r'"([^"]+)"', r'\1', prod)
            # Remove spaces around punctuation and operators
            # Remove space before: ( [ { = , ; :
            cleaned = re.sub(r'\s+([\(\[\{=,;:])', r'\1', cleaned)
            # Remove space after: ) ] } = , ; :
            cleaned = re.sub(r'([\)\]\}=,;:])\s+', r'\1', cleaned)
            # Remove space between word and < (non-terminal)
            cleaned = re.sub(r'(\w)\s+(<)', r'\1\2', cleaned)
            # Remove space between > and word
            cleaned = re.sub(r'(>)\s+(\w)', r'\1\2', cleaned)
            # Remove space between > and (
            cleaned = re.sub(r'(>)\s+(\()', r'\1\2', cleaned)
            # Remove space between ) and <
            cleaned = re.sub(r'(\))\s+(<)', r'\1\2', cleaned)
            # Clean up multiple spaces to single space (for cases where we want to keep one)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned_productions.append(cleaned)
        cleaned_grammar[rule_name] = cleaned_productions
    
    return cleaned_grammar


# Parse BNF grammar into dict format for backward compatibility
GRAMMAR = _parse_bnf_grammar(GRAMMAR_BNF)


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
        terminal_rules = ["number", "string", "split_ratio", "int_number", "score_metric", "model_name",
                         "penalty_string", "kernel_string", "criterion_string", "gamma_value", "small_number",
                         "positive_number", "min_samples_value", "min_samples_ratio", "min_samples_leaf_value",
                         "min_samples_split_int", "min_samples_leaf_ratio", "voting_string"]
        if rule in terminal_rules:
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
    
    # Find all non-terminals in the production and expand them recursively
    import re
    # Find non-terminals in angle brackets: <non_terminal>
    pattern = r'<([^>]+)>'
    
    # Keep expanding until no more non-terminals remain
    max_iterations = 20
    iteration = 0
    while '<' in result and '>' in result and iteration < max_iterations:
        iteration += 1
        def replace_non_terminal(match):
            non_terminal = match.group(1)
            if current_depth < max_depth:
                expanded = generate_dsl_from_grammar(non_terminal, max_depth, current_depth + 1)
                # If expansion failed or still has non-terminals, return as-is for next iteration
                if not expanded or (expanded == non_terminal and non_terminal not in GRAMMAR):
                    return f'<{non_terminal}>'
                return expanded
            return f'<{non_terminal}>'
        
        new_result = re.sub(pattern, replace_non_terminal, result)
        if new_result == result:
            break  # No more changes
        result = new_result
    
    # Handle empty params - if result is empty or just whitespace, return empty
    if result.strip() == "":
        return ""
    
    # Handle special cases: assignments and quoted strings
    # For assignments like "C=<number>", we need to expand "<number>"
    # This needs to be recursive to handle nested non-terminals
    def expand_assignments(text, depth=0, max_assign_depth=5):
        if depth >= max_assign_depth:
            return text  # Prevent infinite recursion
        # Find patterns like "key=<non_terminal>"
        pattern = r'(\w+)=<([^>]+)>'
        def replace_assignment(match):
            key = match.group(1)
            value_rule = match.group(2)
            if value_rule in GRAMMAR and current_depth + depth < max_depth:
                expanded_value = generate_dsl_from_grammar(value_rule, max_depth, current_depth + depth + 1)
                # If expansion still contains non-terminals, recurse
                if expanded_value and '<' in expanded_value and '>' in expanded_value:
                    expanded_value = expand_assignments(expanded_value, depth + 1, max_assign_depth)
                # Handle empty expansion
                if expanded_value == "":
                    return ""
                # Add quotes for string types
                if value_rule in ["string", "penalty_string", "kernel_string", "criterion_string", "score_metric", "voting_string", "model_name"]:
                    if not expanded_value.startswith('"'):
                        return f'{key}="{expanded_value}"'
                return f'{key}={expanded_value}'
            return match.group(0)
        new_text = re.sub(pattern, replace_assignment, text)
        # If we made changes and there are still non-terminals, recurse
        if new_text != text and '<' in new_text and '>' in new_text:
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
    
    # Clean up any remaining angle brackets (shouldn't happen, but just in case)
    # Only remove if the non-terminal is not in grammar (it's a literal)
    def remove_unexpanded(match):
        nt = match.group(1)
        if nt not in GRAMMAR:
            return nt  # It's a literal, remove brackets
        return match.group(0)  # Keep brackets if it's a non-terminal we should have expanded
    
    result = re.sub(r'<([^>]+)>', remove_unexpanded, result)
    
    return result
