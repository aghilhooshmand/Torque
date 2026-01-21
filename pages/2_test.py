"""
Torque DSL - Test Page

Test DSL programs on datasets and view performance metrics.
"""

import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Import modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compiler import dsl_to_sklearn_estimator
from dsl_mapper import map_dsl_to_ast as parse_dsl_to_ast
from evaluator import evaluate_program

st.set_page_config(
    page_title="Torque DSL - Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Test DSL Programs")

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "test_results" not in st.session_state:
    st.session_state.test_results = None

# ============================================
# PART 1: Dataset Section
# ============================================
st.header("Part 1: Dataset")

col_data_left, col_data_right = st.columns([1, 1])

with col_data_left:
    st.subheader("📊 Data Source")
    
    data_source = st.radio(
        "Choose data source",
        ["Upload CSV", "Create Mock Data (Classification)", "Create Mock Data (Regression)"],
        horizontal=False
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                
                st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Select target column
                target_col = st.selectbox(
                    "Select Target Column",
                    df.columns,
                    key="target_select"
                )
                
                # Select feature columns
                feature_cols = st.multiselect(
                    "Select Feature Columns",
                    [c for c in df.columns if c != target_col],
                    default=[c for c in df.columns if c != target_col],
                    key="feature_select"
                )
                
                if feature_cols and st.button("📥 Load Dataset", use_container_width=True):
                    X = df[feature_cols].values
                    y = df[target_col].values
                    
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.target_name = target_col
                    st.session_state.feature_names = feature_cols
                    
                    st.success(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    elif "Create Mock Data" in data_source:
        is_classification = "Classification" in data_source
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Number of Samples", 50, 10000, 200, 50, key="mock_n_samples")
            n_features = st.number_input("Number of Features", 2, 50, 4, 1, key="mock_n_features")
        
        with col2:
            if is_classification:
                n_classes = st.number_input("Number of Classes", 2, 10, 2, 1, key="mock_n_classes")
                # n_informative must be less than n_features (sklearn requirement)
                max_informative = max(2, n_features - 1)  # At least 1 feature must be non-informative
                n_informative = st.number_input(
                    "Informative Features", 
                    2, 
                    max_informative, 
                    min(4, max_informative), 
                    1,
                    key="mock_n_informative",
                    help=f"Must be less than total features ({n_features}). Max: {max_informative}"
                )
            else:
                n_targets = st.number_input("Number of Targets", 1, 3, 1, 1, key="mock_n_targets")
        
        random_state = st.number_input("Random Seed", 0, 1000, 42, 1, key="mock_random_state")
        
        if st.button("🎲 Generate Mock Data", use_container_width=True):
            try:
                if is_classification:
                    # Ensure n_informative is valid (must be < n_features)
                    # sklearn requires: n_informative + n_redundant + n_repeated < n_features
                    n_informative_valid = min(n_informative, n_features - 1)
                    if n_informative_valid < 2:
                        n_informative_valid = max(2, n_features - 1) if n_features > 2 else 2
                    
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=n_classes,
                        n_informative=n_informative_valid,
                        n_redundant=0,  # No redundant features
                        n_repeated=0,   # No repeated features
                        random_state=random_state
                    )
                else:
                    # For now, use classification even for "regression" option
                    # (ensembles work better with classification)
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=3,
                        n_informative=min(4, n_features),
                        random_state=random_state
                    )
                
                # Create DataFrame for display
                feature_names = [f"feature_{i+1}" for i in range(n_features)]
                df = pd.DataFrame(X, columns=feature_names)
                df["target"] = y
                
                st.session_state.dataset = df
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.target_name = "target"
                st.session_state.feature_names = feature_names
                
                st.success(f"✅ Generated: {X.shape[0]} samples, {X.shape[1]} features")
            
            except Exception as e:
                st.error(f"❌ Error generating data: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

with col_data_right:
    st.subheader("📋 Dataset Preview & Info")
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Preview
        st.markdown("**Preview (first 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Info
        st.markdown("**Dataset Information:**")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Rows", df.shape[0])
        with col_info2:
            st.metric("Columns", df.shape[1])
        with col_info3:
            if st.session_state.X is not None:
                st.metric("Features", st.session_state.X.shape[1])
        
        # Target distribution
        if st.session_state.y is not None:
            st.markdown("**Target Distribution:**")
            target_counts = pd.Series(st.session_state.y).value_counts().sort_index()
            st.bar_chart(target_counts)
            
            st.markdown("**Target Statistics:**")
            st.write(f"- Unique values: {len(np.unique(st.session_state.y))}")
            st.write(f"- Min: {st.session_state.y.min()}, Max: {st.session_state.y.max()}")
    else:
        st.info("💡 Upload a CSV file or generate mock data to see preview")

st.markdown("---")

# ============================================
# PART 2: DSL Testing Section
# ============================================
st.header("Part 2: Test DSL Program")

col_dsl_left, col_dsl_right = st.columns([1, 1])

with col_dsl_left:
    st.subheader("✍️ DSL Editor")
    
    # DSL input
    dsl_string = st.text_area(
        "Enter DSL Program",
        value=st.session_state.get("dsl_string", 'vote(LR(C=1.0), SVM(C=1.0, kernel="rbf"); voting="hard")'),
        height=150,
        key="dsl_editor",
        help="Enter your DSL program here"
    )
    st.session_state.dsl_string = dsl_string
    
    # Test settings
    with st.expander("⚙️ Test Settings"):
        test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05, key="test_size_slider")
        random_state = st.number_input("Random Seed", 0, 1000, 42, 1, key="test_random_state")
    
    # Run button
    if st.button("▶️ Run DSL", type="primary", use_container_width=True):
        if st.session_state.X is None or st.session_state.y is None:
            st.error("❌ Please load a dataset first!")
        elif not dsl_string.strip():
            st.error("❌ Please enter a DSL program!")
        else:
            try:
                with st.spinner("Running DSL program..."):
                    # Parse and compile
                    ast = parse_dsl_to_ast(dsl_string)
                    estimator = dsl_to_sklearn_estimator(dsl_string)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X,
                        st.session_state.y,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=st.session_state.y if len(np.unique(st.session_state.y)) > 1 else None
                    )
                    
                    # Fit and predict
                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Store results
                    st.session_state.test_results = {
                        "accuracy": accuracy,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "estimator": estimator,
                        "ast": ast
                    }
                    
                    st.success("✅ DSL program executed successfully!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error running DSL: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

with col_dsl_right:
    st.subheader("📊 Results & Metrics")
    
    if st.session_state.test_results is not None:
        results = st.session_state.test_results
        
        # Metrics
        st.markdown("**Performance Metrics:**")
        
        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
            st.metric("F1 Score", f"{results['f1']:.4f}")
        with col_met2:
            st.metric("Precision", f"{results['precision']:.4f}")
            st.metric("Recall", f"{results['recall']:.4f}")
        
        # Classification report
        st.markdown("**Detailed Classification Report:**")
        report = classification_report(
            results['y_test'],
            results['y_pred'],
            output_dict=True,
            zero_division=0
        )
        st.json(report)
        
        # Confusion matrix
        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        st.dataframe(pd.DataFrame(cm), use_container_width=True)
        
        # Model info
        with st.expander("🔍 Model Information"):
            st.write(f"**Type:** {type(results['estimator']).__name__}")
            if hasattr(results['estimator'], 'get_params'):
                st.write("**Parameters:**")
                params = results['estimator'].get_params(deep=True)
                st.json(params)
        
        # AST info
        with st.expander("🌳 AST Structure"):
            st.json(results['ast'])
    
    else:
        st.info("💡 Enter a DSL program and click 'Run DSL' to see results")

st.markdown("---")

# Footer
st.markdown("**💡 Tip:** Use the Grammar page to generate DSL strings, then test them here!")

