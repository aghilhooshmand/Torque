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
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Import modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset_loader import build_X_y_from_frames, fetch_uci_data, sample_stratified_by_class
from Torque_runner import run_dsl

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
if "uci_features" not in st.session_state:
    st.session_state.uci_features = None
if "uci_targets" not in st.session_state:
    st.session_state.uci_targets = None

# ============================================
# PART 1: Dataset Section
# ============================================
st.header("Part 1: Dataset")

col_data_left, col_data_right = st.columns([1, 1])

with col_data_left:
    st.subheader("📊 Data Source")
    
    data_source = st.radio(
        "Choose data source",
        [
            "Upload CSV",
            "Fetch from UCI ML Repository",
            "Create Mock Data (Classification)",
            "Create Mock Data (Regression)",
        ],
        horizontal=False,
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

                pct_per_class_upload = st.slider(
                    "Percent of each class to use",
                    min_value=1,
                    max_value=100,
                    value=100,
                    step=1,
                    key="upload_pct_slider",
                    help="Use this % of each class to reduce dataset size while keeping class balance (stratified sample).",
                )
                upload_random_seed = st.number_input(
                    "Random seed for sampling",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    key="upload_sample_seed",
                )

                if feature_cols and st.button("📥 Load Dataset", use_container_width=True):
                    X = df[feature_cols].values
                    y = df[target_col].values

                    if pct_per_class_upload < 100:
                        X, y, kept_idx = sample_stratified_by_class(
                            X, y, float(pct_per_class_upload), random_state=int(upload_random_seed)
                        )
                        df_preview = df[feature_cols + [target_col]].iloc[kept_idx].reset_index(drop=True)
                    else:
                        df_preview = df[feature_cols + [target_col]].copy()

                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.target_name = target_col
                    st.session_state.feature_names = feature_cols
                    st.session_state.dataset = df_preview
                    
                    st.success(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    elif data_source == "Fetch from UCI ML Repository":
        uci_id = st.number_input(
            "UCI Dataset ID",
            min_value=1,
            max_value=1000,
            value=17,
            step=1,
            key="uci_id_input",
            help="Numeric ID of the dataset on UCI ML Repository (e.g. 2 = Adult, 17 = Breast Cancer Wisconsin, 31 = Credit-g).",
        )
        if st.button("🔍 Fetch from UCI", use_container_width=True, key="uci_fetch_btn"):
            try:
                with st.spinner("Fetching dataset from UCI ML Repository..."):
                    X_df, y_df = fetch_uci_data(int(uci_id))
                st.session_state.uci_features = X_df
                st.session_state.uci_targets = y_df
                st.success(f"✅ Fetched: {X_df.shape[0]} rows, {X_df.shape[1]} features, {y_df.shape[1]} target column(s)")
            except ImportError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Error fetching UCI dataset: {e}")

        if st.session_state.uci_features is not None and st.session_state.uci_targets is not None:
            X_df = st.session_state.uci_features
            y_df = st.session_state.uci_targets
            target_col = st.selectbox(
                "Select Target Column",
                list(y_df.columns),
                key="uci_target_select",
            )
            feature_cols = st.multiselect(
                "Select Feature Columns",
                list(X_df.columns),
                default=list(X_df.columns),
                key="uci_feature_select",
            )
            pct_per_class = st.slider(
                "Percent of each class to use",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                key="uci_pct_slider",
                help="Use this % of each class to reduce dataset size while keeping class balance (stratified sample).",
            )
            uci_random_seed = st.number_input(
                "Random seed for sampling",
                min_value=0,
                max_value=99999,
                value=42,
                key="uci_sample_seed",
            )
            if feature_cols and st.button("📥 Load Dataset", key="uci_load_btn", use_container_width=True):
                try:
                    X, y = build_X_y_from_frames(X_df, y_df, target_col, feature_cols)
                    if pct_per_class < 100:
                        X, y, kept_idx = sample_stratified_by_class(
                            X, y, float(pct_per_class), random_state=int(uci_random_seed)
                        )
                    else:
                        kept_idx = np.arange(len(y))
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.target_name = target_col
                    st.session_state.feature_names = feature_cols
                    # Build a single DataFrame for preview (same rows as X, y)
                    df_preview = X_df[feature_cols].copy()
                    df_preview[target_col] = y_df[target_col].values
                    df_preview = df_preview.iloc[kept_idx].reset_index(drop=True)
                    st.session_state.dataset = df_preview
                    st.success(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {e}")

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

        # Optional downsampling per class for mock classification data
        if is_classification:
            pct_per_class_mock = st.slider(
                "Percent of each class to use",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                key="mock_pct_slider",
                help="Use this % of each class to reduce dataset size while keeping class balance (stratified sample).",
            )
            mock_sample_seed = st.number_input(
                "Random seed for sampling (mock)",
                min_value=0,
                max_value=99999,
                value=42,
                key="mock_sample_seed",
            )
        else:
            pct_per_class_mock = 100
            mock_sample_seed = int(random_state)
        
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

                # Optional stratified downsampling
                if pct_per_class_mock < 100:
                    X, y, kept_idx = sample_stratified_by_class(
                        X, y, float(pct_per_class_mock), random_state=int(mock_sample_seed)
                    )
                else:
                    kept_idx = np.arange(len(y))

                # Create DataFrame for display (only kept rows)
                feature_names = [f"feature_{i+1}" for i in range(n_features)]
                df = pd.DataFrame(X, columns=feature_names)
                df["target"] = y
                df = df.reset_index(drop=True)
                
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
        st.info("💡 Upload a CSV, fetch from UCI ML Repository, or generate mock data to see preview")

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
        value=st.session_state.get("dsl_string", 'vote(LR(C=1.0), DT(max_depth=5); voting="hard")'),
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
                with st.spinner("Running Torque DSL via Torque_runner..."):
                    results_dir = os.path.join(current_dir, "results")
                    os.makedirs(results_dir, exist_ok=True)
                    results = run_dsl(
                        torque_command=dsl_string,
                        X=st.session_state.X,
                        y=st.session_state.y,
                        test_size=float(test_size),
                        random_state=int(random_state),
                        mapped_json_file=os.path.join(results_dir, "Torque_mapper_result.json"),
                        metrics_json_file=os.path.join(results_dir, "Torque_runner_result.json"),
                    )
                    st.session_state.test_results = results
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
        metrics = results.get("metrics", {})
        
        # Metrics
        st.markdown("**Performance Metrics (from Torque_runner):**")
        
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            if "accuracy" in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            if "f1_macro" in metrics:
                st.metric("F1 (macro)", f"{metrics['f1_macro']:.4f}")
        with col_met2:
            if "precision_macro" in metrics:
                st.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
            if "recall_macro" in metrics:
                st.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")
        with col_met3:
            if "roc_auc" in metrics:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        # Classification report
        class_report = results.get("classification_report")
        if class_report:
            st.markdown("**Detailed Classification Report:**")
            st.json(class_report)
        
        # Confusion matrix
        cm_info = results.get("confusion_matrix")
        if cm_info:
            st.markdown("**Confusion Matrix:**")
            cm = np.array(cm_info.get("matrix", []))
            labels = cm_info.get("labels", [])
            if cm.size > 0:
                df_cm = pd.DataFrame(cm, index=labels, columns=labels)
                st.dataframe(df_cm, use_container_width=True)
        
        # Model info
        est_info = results.get("estimator_info", {})
        with st.expander("🔍 Model Information"):
            if est_info:
                st.json(est_info)
            source = results.get("source", {})
            mapping_file = results.get("mapping_file")
            if source:
                st.markdown("**Source info:**")
                st.json(source)
            if mapping_file:
                st.markdown(f"- Mapping file: `{mapping_file}`")
    
    else:
        st.info("💡 Enter a DSL program and click 'Run DSL' to see results")

st.markdown("---")

# Footer
st.markdown("**💡 Tip:** Use the Grammar page to generate DSL strings, then test them here!")

