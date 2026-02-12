"""
Torque DSL - Meta-Feature Explorer

Upload / fetch / mock datasets and compute meta-features using PyMFE and
custom measures (intrinsic dimensionality, basic stats). Also compare
meta-features across two datasets.
"""

import os
import sys
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st
from pymfe.mfe import MFE
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import skdim
except ImportError:  # pragma: no cover - optional dependency
    skdim = None


def _compute_basic_stats(X: np.ndarray, y: np.ndarray, df: Union[pd.DataFrame, None] = None) -> Dict[str, float]:
    """
    Compute simple, interpretable dataset-level statistics from the actual data
    used for meta-feature extraction (X, y). Size stats always come from X/y so
    they match the stored X_shape. Optionally use df for missing/outliers only if
    it has the same number of rows as X.
    """
    stats: Dict[str, float] = {}
    n_rows = int(X.shape[0])
    n_features_only = int(X.shape[1])
    stats["n_rows"] = float(n_rows)
    stats["n_features_only"] = float(n_features_only)
    stats["n_columns"] = float(n_features_only + 1)  # features + target

    # Missing / outliers from df only if it matches the actual data size
    if df is not None and len(df) == n_rows:
        n_missing = int(df.isna().sum().sum())
        stats["missing_total"] = float(n_missing)
        n_cells = n_rows * df.shape[1]
        stats["missing_pct"] = float(n_missing / n_cells) * 100.0 if n_cells > 0 else 0.0
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            z = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0).replace(0, np.nan)
            is_outlier = (np.abs(z) > 3).any(axis=1)
            stats["outliers_count_z3"] = float(is_outlier.sum())
            stats["outliers_pct_z3"] = float(is_outlier.mean()) * 100.0
        else:
            stats["outliers_count_z3"] = 0.0
            stats["outliers_pct_z3"] = 0.0
    else:
        # No matching df: missing/outliers from X (often no NaNs in numeric X)
        stats["missing_total"] = float(np.isnan(X).sum())
        n_cells = X.size
        stats["missing_pct"] = float(np.isnan(X).sum() / n_cells) * 100.0 if n_cells > 0 else 0.0
        try:
            x_mean = np.nanmean(X, axis=0)
            x_std = np.nanstd(X, axis=0, ddof=0)
            x_std[x_std == 0] = np.nan
            z = (X - x_mean) / x_std
            is_outlier = (np.abs(z) > 3).any(axis=1)
            stats["outliers_count_z3"] = float(is_outlier.sum())
            stats["outliers_pct_z3"] = float(is_outlier.mean()) * 100.0
        except Exception:
            stats["outliers_count_z3"] = 0.0
            stats["outliers_pct_z3"] = 0.0

    # Class imbalance from y (always matches X row count)
    if y is not None:
        y_arr = np.asarray(y).ravel()
        unique_vals, counts = np.unique(y_arr, return_counts=True)
        if len(unique_vals) > 1 and len(unique_vals) <= 50:
            stats["n_classes"] = float(len(unique_vals))
            max_c = counts.max()
            min_c = counts.min()
            stats["class_imbalance_ratio_max_min"] = float(max_c / min_c) if min_c > 0 else float("inf")
        else:
            stats["n_classes"] = float(len(unique_vals))

    return stats

# Import project modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset_loader import build_X_y_from_frames, fetch_uci_data, sample_stratified_by_class

# PyMFE complexity + landmarking features that best indicate dataset difficulty (easier vs harder).
# See META_FEATURES_GUIDE: l1–l3 (linearity), n1–n4 (neighborhood), f1–f4 (feature overlap), landmarking (lda, NB, 1NN, dtree).
COMPLEXITY_SUMMARY_FACTOR_BASES = frozenset({
    "l1", "l2", "l3", "n1", "n2", "n3", "n4", "f1", "f2", "f3", "f4",
    "linear_discr", "naive_bayes", "one_nn", "best_node",
})

# For complexity_summary_factor: "lower" = lower value = easier (hardness indicators); "higher" = higher value = easier (ease indicators).
# Base names only; PyMFE may return e.g. "l2.mean" -> base "l2".
COMPLEXITY_LOWER_IS_EASIER = frozenset({"l2", "l3", "n1", "n2", "n3", "n4", "f1", "f2", "f4"})  # hardness: lower = easier
COMPLEXITY_HIGHER_IS_EASIER = frozenset({"l1", "f3", "linear_discr", "naive_bayes", "one_nn", "best_node"})  # ease: higher = easier


st.set_page_config(
    page_title="Torque DSL - Meta-Features",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Meta-Feature Explorer")
st.markdown(
    "Compute **dataset meta-features** (via PyMFE) for uploaded, UCI, or mock datasets, "
    "then **compare meta-features between datasets**."
)

# Link to guide
col_guide_left, col_guide_right = st.columns([3, 1])
with col_guide_left:
    st.info(
        "💡 **Quick Start**: Compute `basic-stats` + `pca-structure` + `intrinsic-dim` for quick complexity check. "
        "Key measures: `pca_k95_over_n_features` (< 0.5 = simpler), `id_mle` (lower = simpler), "
        "`class_imbalance_ratio_max_min` (< 10 = balanced)."
    )
with col_guide_right:
    if st.button("📖 Open Full Guide", use_container_width=True, type="secondary"):
        st.switch_page("pages/5_guide.py")


# ============================================
# Session state
# ============================================
if "mf_dataset_df" not in st.session_state:
    st.session_state.mf_dataset_df = None
if "mf_X" not in st.session_state:
    st.session_state.mf_X = None
if "mf_y" not in st.session_state:
    st.session_state.mf_y = None
if "mf_target_name" not in st.session_state:
    st.session_state.mf_target_name = None
if "mf_feature_names" not in st.session_state:
    st.session_state.mf_feature_names = None
if "mf_uci_features" not in st.session_state:
    st.session_state.mf_uci_features = None
if "mf_uci_targets" not in st.session_state:
    st.session_state.mf_uci_targets = None
# Stored datasets with meta-features: {label: {"X_shape": ..., "y_shape": ..., "meta": {group: {name: value}}}}
if "mf_store" not in st.session_state:
    st.session_state.mf_store: Dict[str, Dict] = {}


st.markdown("---")

# ============================================
# PART 1: Dataset selection / creation
# ============================================
st.header("Part 1: Build a Dataset")

col_left, col_right = st.columns([1, 1])

with col_left:
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
        key="mf_data_source_radio",
    )

    # ---------- Upload CSV ----------
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="mf_upload_csv")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.mf_dataset_df = df

                st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

                target_col = st.selectbox(
                    "Select Target Column",
                    df.columns,
                    key="mf_target_select",
                )

                feature_cols = st.multiselect(
                    "Select Feature Columns",
                    [c for c in df.columns if c != target_col],
                    default=[c for c in df.columns if c != target_col],
                    key="mf_feature_select",
                )

                pct_per_class_upload = st.slider(
                    "Percent of each class to use",
                    min_value=1,
                    max_value=100,
                    value=100,
                    step=1,
                    key="mf_upload_pct_slider",
                    help=(
                        "Use this % of each class to reduce dataset size while keeping "
                        "class balance (stratified sample)."
                    ),
                )
                upload_random_seed = st.number_input(
                    "Random seed for sampling",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    key="mf_upload_sample_seed",
                )

                if feature_cols and st.button("📥 Load Dataset", use_container_width=True, key="mf_upload_load_btn"):
                    X = df[feature_cols].values
                    y = df[target_col].values

                    if pct_per_class_upload < 100:
                        X, y, kept_idx = sample_stratified_by_class(
                            X,
                            y,
                            float(pct_per_class_upload),
                            random_state=int(upload_random_seed),
                        )
                        df_preview = df[feature_cols + [target_col]].iloc[kept_idx].reset_index(drop=True)
                    else:
                        df_preview = df[feature_cols + [target_col]].copy()

                    st.session_state.mf_X = X
                    st.session_state.mf_y = y
                    st.session_state.mf_target_name = target_col
                    st.session_state.mf_feature_names = feature_cols
                    st.session_state.mf_dataset_df = df_preview

                    st.success(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    # ---------- UCI ----------
    elif data_source == "Fetch from UCI ML Repository":
        uci_id = st.number_input(
            "UCI Dataset ID",
            min_value=1,
            max_value=1000,
            value=17,
            step=1,
            key="mf_uci_id_input",
            help=(
                "Numeric ID of the dataset on UCI ML Repository "
                "(e.g. 2 = Adult, 17 = Breast Cancer Wisconsin, 31 = Credit-g)."
            ),
        )
        if st.button("🔍 Fetch from UCI", use_container_width=True, key="mf_uci_fetch_btn"):
            try:
                with st.spinner("Fetching dataset from UCI ML Repository..."):
                    X_df, y_df = fetch_uci_data(int(uci_id))
                st.session_state.mf_uci_features = X_df
                st.session_state.mf_uci_targets = y_df
                st.success(f"✅ Fetched: {X_df.shape[0]} rows, {X_df.shape[1]} features, {y_df.shape[1]} target column(s)")
            except ImportError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Error fetching UCI dataset: {e}")

        if st.session_state.mf_uci_features is not None and st.session_state.mf_uci_targets is not None:
            X_df = st.session_state.mf_uci_features
            y_df = st.session_state.mf_uci_targets
            target_col = st.selectbox(
                "Select Target Column",
                list(y_df.columns),
                key="mf_uci_target_select",
            )
            feature_cols = st.multiselect(
                "Select Feature Columns",
                list(X_df.columns),
                default=list(X_df.columns),
                key="mf_uci_feature_select",
            )
            pct_per_class = st.slider(
                "Percent of each class to use",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                key="mf_uci_pct_slider",
                help=(
                    "Use this % of each class to reduce dataset size while keeping "
                    "class balance (stratified sample)."
                ),
            )
            uci_random_seed = st.number_input(
                "Random seed for sampling",
                min_value=0,
                max_value=99999,
                value=42,
                key="mf_uci_sample_seed",
            )
            if feature_cols and st.button("📥 Load Dataset", key="mf_uci_load_btn", use_container_width=True):
                try:
                    X, y = build_X_y_from_frames(X_df, y_df, target_col, feature_cols)
                    if pct_per_class < 100:
                        X, y, kept_idx = sample_stratified_by_class(
                            X,
                            y,
                            float(pct_per_class),
                            random_state=int(uci_random_seed),
                        )
                    else:
                        kept_idx = np.arange(len(y))
                    st.session_state.mf_X = X
                    st.session_state.mf_y = y
                    st.session_state.mf_target_name = target_col
                    st.session_state.mf_feature_names = feature_cols

                    df_preview = X_df[feature_cols].copy()
                    df_preview[target_col] = y_df[target_col].values
                    df_preview = df_preview.iloc[kept_idx].reset_index(drop=True)
                    st.session_state.mf_dataset_df = df_preview

                    st.success(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {e}")

    # ---------- Mock data ----------
    elif "Create Mock Data" in data_source:
        is_classification = "Classification" in data_source

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input(
                "Number of Samples",
                50,
                10000,
                200,
                50,
                key="mf_mock_n_samples",
            )
            n_features = st.number_input(
                "Number of Features",
                2,
                50,
                4,
                1,
                key="mf_mock_n_features",
            )

        with col2:
            if is_classification:
                n_classes = st.number_input(
                    "Number of Classes",
                    2,
                    10,
                    2,
                    1,
                    key="mf_mock_n_classes",
                )
                max_informative = max(2, n_features - 1)
                n_informative = st.number_input(
                    "Informative Features",
                    2,
                    max_informative,
                    min(4, max_informative),
                    1,
                    key="mf_mock_n_informative",
                    help=f"Must be less than total features ({n_features}). Max: {max_informative}",
                )
            else:
                _ = st.number_input(
                    "Number of Targets (for regression)",
                    1,
                    3,
                    1,
                    1,
                    key="mf_mock_n_targets",
                )

        random_state = st.number_input(
            "Random Seed",
            0,
            1000,
            42,
            1,
            key="mf_mock_random_state",
        )

        if is_classification:
            pct_per_class_mock = st.slider(
                "Percent of each class to use",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                key="mf_mock_pct_slider",
                help=(
                    "Use this % of each class to reduce dataset size while keeping "
                    "class balance (stratified sample)."
                ),
            )
            mock_sample_seed = st.number_input(
                "Random seed for sampling (mock)",
                min_value=0,
                max_value=99999,
                value=42,
                key="mf_mock_sample_seed",
            )
        else:
            pct_per_class_mock = 100
            mock_sample_seed = int(random_state)

        if st.button("🎲 Generate Mock Data", use_container_width=True, key="mf_mock_generate_btn"):
            try:
                if is_classification:
                    n_informative_valid = min(n_informative, n_features - 1)
                    if n_informative_valid < 2:
                        n_informative_valid = max(2, n_features - 1) if n_features > 2 else 2

                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=n_classes,
                        n_informative=n_informative_valid,
                        n_redundant=0,
                        n_repeated=0,
                        random_state=random_state,
                    )
                else:
                    # For regression, we still keep a simple regression generator
                    X, y = make_regression(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_targets=1,
                        noise=0.1,
                        random_state=random_state,
                    )

                if is_classification and pct_per_class_mock < 100:
                    X, y, kept_idx = sample_stratified_by_class(
                        X,
                        y,
                        float(pct_per_class_mock),
                        random_state=int(mock_sample_seed),
                    )
                else:
                    kept_idx = np.arange(len(y))

                feature_names = [f"feature_{i+1}" for i in range(n_features)]
                df = pd.DataFrame(X, columns=feature_names)
                df["target"] = y
                df = df.iloc[kept_idx].reset_index(drop=True)

                st.session_state.mf_dataset_df = df
                st.session_state.mf_X = X[kept_idx]
                st.session_state.mf_y = np.asarray(y)[kept_idx]
                st.session_state.mf_target_name = "target"
                st.session_state.mf_feature_names = feature_names

                st.success(f"✅ Generated: {st.session_state.mf_X.shape[0]} samples, {st.session_state.mf_X.shape[1]} features")

            except Exception as e:
                st.error(f"❌ Error generating data: {e}")
                import traceback

                st.code(traceback.format_exc(), language="python")


with col_right:
    st.subheader("📋 Dataset Preview & Info")

    if st.session_state.mf_dataset_df is not None:
        df = st.session_state.mf_dataset_df

        st.markdown("**Preview (first 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("**Dataset Information:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            if st.session_state.mf_X is not None:
                st.metric("Features", st.session_state.mf_X.shape[1])

        if st.session_state.mf_y is not None:
            st.markdown("**Target Distribution (if classification):**")
            y_vals = st.session_state.mf_y
            if np.issubdtype(y_vals.dtype, np.integer) or len(np.unique(y_vals)) < 30:
                target_counts = pd.Series(y_vals).value_counts().sort_index()
                st.bar_chart(target_counts)
    else:
        st.info("💡 Load or generate a dataset to preview it here.")


st.markdown("---")

# ============================================
# PART 2: Meta-feature extraction (PyMFE)
# ============================================
st.header("Part 2: Extract Meta-Features")

if st.session_state.mf_X is None or st.session_state.mf_y is None:
    st.warning("⚠️ Please prepare a dataset in Part 1 before extracting meta-features.")
else:
    group_options: List[str] = [
        "general",
        "statistical",
        "info-theory",
        "model-based",
        "landmarking",
        "complexity_summary_factor",
    ]

    # Group descriptions for help tooltip
    group_help = {
        "general": "Simple dataset characteristics: size, ratios, variable types (nr_inst, nr_attr, inst_to_attr, nr_class). Quick baseline measures.",
        "statistical": "Distributional properties: central tendency, spread, shape, correlations (mean, sd, skewness, kurtosis, cov.mean).",
        "info-theory": "Entropy & mutual information: uncertainty and attribute-class relationships (attr_ent, class_ent, mut_inf). Higher mut_inf = better predictive power.",
        "model-based": "Decision tree complexity: leaves, nodes, purity measures. Higher values indicate more complex decision boundaries.",
        "landmarking": "Simple learner performance (1-NN, Naive Bayes, Linear Discriminant accuracy). Higher scores = easier dataset.",
        "complexity_summary_factor": "Curated complexity indicators: linearity (l1–l3), neighborhood (n1–n4), feature overlap (f1–f4), landmarking (lda, naive_bayes, one_nn, best_node). Easiest vs hardest signal; see Guide.",
    }
    
    selected_groups = st.multiselect(
        "Select meta-feature groups (PyMFE)",
        options=group_options,
        default=group_options,
        help=(
            "Meta-features will be computed separately for each selected group. "
            "💡 Tip: Start with 'general' + 'statistical' for quick assessment. "
            "See guide (📖 above) for detailed descriptions."
        ),
        key="mf_group_multiselect",
    )
    
    # Show descriptions for selected groups
    if selected_groups:
        with st.expander("ℹ️ Selected Groups Description", expanded=False):
            for group in selected_groups:
                if group in group_help:
                    st.markdown(f"**{group}**: {group_help[group]}")

    default_label = st.text_input(
        "Dataset label (for storing & comparison)",
        value="dataset_1" if not st.session_state.mf_store else f"dataset_{len(st.session_state.mf_store) + 1}",
        key="mf_dataset_label",
        help="Name used to store this dataset and its meta-features for later comparison.",
    )

    compute_id = st.checkbox(
        "Also compute intrinsic dimensionality (MLE & TwoNN via scikit-dimension)",
        value=True,
        help=(
            "Estimates 'true degrees of freedom' using geometric methods. "
            "Lower ID (< n_features/2) indicates simpler, low-dimensional manifold. "
            "Compare to pca_k95: if ID < k95, data has non-linear structure."
        ),
        key="mf_compute_id_checkbox",
    )
    
    # Info about automatically computed groups
    st.info(
        "💡 **Always computed**: `basic-stats` (size, missing, imbalance, outliers) and `pca-structure` "
        "(components for 95%/99% variance). These provide quick complexity indicators."
    )

    if st.button("📈 Compute Meta-Features & Store Dataset", type="primary", use_container_width=True):
        if not selected_groups:
            st.error("❌ Please select at least one meta-feature group.")
        elif not default_label.strip():
            st.error("❌ Please enter a non-empty dataset label.")
        elif default_label.strip() in st.session_state.mf_store:
            st.error("❌ A dataset with this label already exists. Please choose another label.")
        else:
            X = st.session_state.mf_X
            y = st.session_state.mf_y
            label = default_label.strip()

            try:
                meta_by_group: Dict[str, Dict[str, float]] = {}
                with st.spinner("Computing meta-features (this may take a few seconds)..."):
                    # --- PyMFE groups ---
                    for group in selected_groups:
                        if group == "complexity_summary_factor":
                            mfe = MFE(groups=["complexity", "landmarking"])
                            mfe.fit(X, y)
                            names, values = mfe.extract()
                            # Keep only complexity-summary features (base name before first '.' in allowed set)
                            filtered = {}
                            for name, val in zip(names, values):
                                base = name.split(".")[0] if "." in name else name
                                if base in COMPLEXITY_SUMMARY_FACTOR_BASES:
                                    try:
                                        filtered[name] = float(val)
                                    except (TypeError, ValueError):
                                        pass
                            meta_by_group["complexity_summary_factor"] = filtered
                        else:
                            mfe = MFE(groups=[group])
                            mfe.fit(X, y)
                            names, values = mfe.extract()
                            meta_by_group[group] = {name: float(val) for name, val in zip(names, values)}

                    # --- Basic stats: from actual X/y so n_rows matches title and Quick Summary ---
                    df_for_stats = st.session_state.mf_dataset_df if (st.session_state.mf_dataset_df is not None and len(st.session_state.mf_dataset_df) == X.shape[0]) else None
                    basic_stats = _compute_basic_stats(X, y, df_for_stats)
                    meta_by_group["basic-stats"] = basic_stats

                    # --- Intrinsic dimensionality (optional) ---
                    if compute_id:
                        if skdim is None:
                            st.warning(
                                "⚠️ Intrinsic dimensionality requested, but 'scikit-dimension' is not installed. "
                                "Run: pip install scikit-dimension"
                            )
                        else:
                            try:
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                id_mle = float(skdim.id.MLE().fit(X_scaled).dimension_)
                                id_twonn = float(skdim.id.TwoNN().fit(X_scaled).dimension_)
                                meta_by_group["intrinsic-dim"] = {
                                    "id_mle": id_mle,
                                    "id_twonn": id_twonn,
                                }
                            except Exception as e_id:
                                st.warning(f"⚠️ Could not compute intrinsic dimensionality: {e_id}")

                    # --- PCA structure: how many dimensions matter (k95 / k99) ---
                    try:
                        scaler_pca = StandardScaler(with_mean=True, with_std=True)
                        X_scaled_pca = scaler_pca.fit_transform(X)
                        pca = PCA()
                        pca.fit(X_scaled_pca)
                        if pca.explained_variance_ratio_.size > 0:
                            cum_var = np.cumsum(pca.explained_variance_ratio_)
                            k95 = int(np.searchsorted(cum_var, 0.95) + 1)
                            k99 = int(np.searchsorted(cum_var, 0.99) + 1)
                            n_features = X.shape[1]
                            meta_by_group["pca-structure"] = {
                                "pca_k95": float(k95),
                                "pca_k99": float(k99),
                                "pca_k95_over_n_features": float(k95) / float(n_features) if n_features > 0 else np.nan,
                                "pca_k99_over_n_features": float(k99) / float(n_features) if n_features > 0 else np.nan,
                            }
                    except Exception as e_pca:
                        st.warning(f"⚠️ Could not compute PCA-based measures: {e_pca}")

                groups_stored = list(selected_groups)
                if "basic-stats" in meta_by_group and "basic-stats" not in groups_stored:
                    groups_stored.append("basic-stats")
                if compute_id and skdim is not None and "intrinsic-dim" in meta_by_group and "intrinsic-dim" not in groups_stored:
                    groups_stored.append("intrinsic-dim")
                if "pca-structure" in meta_by_group and "pca-structure" not in groups_stored:
                    groups_stored.append("pca-structure")

                st.session_state.mf_store[label] = {
                    "X_shape": X.shape,
                    "y_shape": np.asarray(y).shape,
                    "target_name": st.session_state.mf_target_name,
                    "feature_names": list(st.session_state.mf_feature_names or []),
                    "groups": groups_stored,
                    "meta": meta_by_group,
                }

                st.success(
                    f"✅ Stored dataset '{label}' with meta-features for groups: {', '.join(groups_stored)}"
                )

            except Exception as e:
                st.error(f"❌ Error computing meta-features: {e}")
                import traceback

                st.code(traceback.format_exc(), language="python")

if st.session_state.mf_store:
    st.markdown("---")
    st.subheader("📂 Stored Datasets & Meta-Features")

    # Group descriptions for display
    group_display_names = {
        "general": "General: Size, ratios, variable types",
        "statistical": "Statistical: Distribution properties",
        "info-theory": "Info-Theory: Entropy & mutual information",
        "model-based": "Model-Based: Decision tree complexity",
        "landmarking": "Landmarking: Simple learner performance",
        "complexity_summary_factor": "Complexity Summary: l1–l3, n1–n4, f1–f4, landmarking (easier vs harder)",
        "basic-stats": "Basic Stats: Size, missing, imbalance, outliers",
        "pca-structure": "PCA Structure: Components for 95%/99% variance",
        "intrinsic-dim": "Intrinsic Dim: True degrees of freedom (geometric)",
    }

    for label, info in st.session_state.mf_store.items():
        with st.expander(f"📁 {label}  —  {info['X_shape'][0]} samples, {info['X_shape'][1]} features"):
            st.markdown(
                f"- **Target column**: `{info.get('target_name')}`  \n"
                f"- **Meta-feature groups**: {', '.join(info.get('groups', []))}"
            )
            
            # Quick summary for key measures
            quick_measures = {}
            if "basic-stats" in info.get("meta", {}):
                bs = info["meta"]["basic-stats"]
                quick_measures["Rows"] = bs.get("n_rows", "N/A")
                quick_measures["Features"] = bs.get("n_features_only", "N/A")
                quick_measures["Classes"] = bs.get("n_classes", "N/A")
                if "class_imbalance_ratio_max_min" in bs:
                    quick_measures["Imbalance Ratio"] = f"{bs['class_imbalance_ratio_max_min']:.2f}"
            if "pca-structure" in info.get("meta", {}):
                ps = info["meta"]["pca-structure"]
                if "pca_k95_over_n_features" in ps:
                    quick_measures["PCA k95/n_features"] = f"{ps['pca_k95_over_n_features']:.3f}"
            if "intrinsic-dim" in info.get("meta", {}):
                id_meta = info["meta"]["intrinsic-dim"]
                if "id_mle" in id_meta:
                    quick_measures["ID (MLE)"] = f"{id_meta['id_mle']:.2f}"
            
            if quick_measures:
                st.markdown("**Quick Summary:**")
                cols = st.columns(len(quick_measures))
                for i, (k, v) in enumerate(quick_measures.items()):
                    with cols[i]:
                        st.metric(k, v)
                st.markdown("---")
            
            # Detailed tables per group
            for group in info.get("meta", {}):
                display_name = group_display_names.get(group, group)
                st.markdown(f"**{display_name}**")
                meta_dict = info["meta"][group]
                if meta_dict:
                    df_meta = pd.DataFrame(
                        {
                            "meta_feature": list(meta_dict.keys()),
                            "value": list(meta_dict.values()),
                        }
                    ).sort_values("meta_feature")
                    st.dataframe(df_meta, use_container_width=True, hide_index=True)
                    
                    # Add interpretation hints for key measures
                    if group == "pca-structure" and "pca_k95_over_n_features" in meta_dict:
                        k95_ratio = meta_dict["pca_k95_over_n_features"]
                        if k95_ratio < 0.5:
                            st.success(f"✓ Low redundancy: k95/n_features = {k95_ratio:.3f} (< 0.5)")
                        elif k95_ratio > 0.8:
                            st.warning(f"⚠ High complexity: k95/n_features = {k95_ratio:.3f} (> 0.8)")
                    elif group == "basic-stats" and "class_imbalance_ratio_max_min" in meta_dict:
                        imb_ratio = meta_dict["class_imbalance_ratio_max_min"]
                        if imb_ratio > 10:
                            st.warning(f"⚠ Severe class imbalance: ratio = {imb_ratio:.2f} (> 10)")
                        elif imb_ratio < 2:
                            st.success(f"✓ Balanced classes: ratio = {imb_ratio:.2f}")
                else:
                    st.info("No meta-features extracted for this group.")
                st.markdown("")  # spacing


st.markdown("---")

# ============================================
# PART 3: Compare meta-features across datasets
# ============================================
st.header("Part 3: Compare Datasets")

if not st.session_state.mf_store or len(st.session_state.mf_store) < 2:
    st.info("💡 Store at least **two** datasets with meta-features to enable comparison.")
else:
    labels = list(st.session_state.mf_store.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        ds_a = st.selectbox("Dataset A", labels, index=0, key="mf_compare_a")
    with col_b:
        ds_b = st.selectbox("Dataset B", labels, index=1, key="mf_compare_b")

    common_groups = sorted(
        set(st.session_state.mf_store[ds_a]["meta"].keys())
        & set(st.session_state.mf_store[ds_b]["meta"].keys())
    )
    if not common_groups:
        st.warning("⚠️ Selected datasets do not share any meta-feature groups. Recompute with overlapping groups.")
    else:
        group = st.selectbox("Meta-feature group to compare", common_groups, key="mf_compare_group")

        meta_a = st.session_state.mf_store[ds_a]["meta"].get(group, {})
        meta_b = st.session_state.mf_store[ds_b]["meta"].get(group, {})

        all_features = sorted(set(meta_a.keys()) | set(meta_b.keys()))
        is_complexity_summary = group == "complexity_summary_factor"

        rows = []
        for name in all_features:
            va = meta_a.get(name, np.nan)
            vb = meta_b.get(name, np.nan)
            diff = np.nan
            if not (np.isnan(va) or np.isnan(vb)):
                diff = vb - va
            # B vs A indicator: ↑ B higher, ↓ B lower, — same
            if np.isnan(diff):
                vs = "—"
            elif diff > 0:
                vs = "↑ B higher"
            elif diff < 0:
                vs = "↓ B lower"
            else:
                vs = "—"

            # For complexity_summary_factor: which dataset is easier (less complex)?
            easier = "—"
            if is_complexity_summary and not (np.isnan(va) or np.isnan(vb)):
                base = name.split(".")[0] if "." in name else name
                if base in COMPLEXITY_LOWER_IS_EASIER:
                    # Hardness: lower value = easier
                    if vb < va:
                        easier = "B"
                    elif vb > va:
                        easier = "A"
                    else:
                        easier = "Same"
                elif base in COMPLEXITY_HIGHER_IS_EASIER:
                    # Ease: higher value = easier
                    if vb > va:
                        easier = "B"
                    elif vb < va:
                        easier = "A"
                    else:
                        easier = "Same"

            row = {
                "meta_feature": name,
                f"{ds_a}": va,
                f"{ds_b}": vb,
                "diff (B − A)": diff,
                "B vs A": vs,
            }
            if is_complexity_summary:
                row["Easier"] = easier
            rows.append(row)

        df_compare = pd.DataFrame(rows).sort_values("meta_feature")

        def _cell_style_diff(val):
            if pd.isna(val):
                return ""
            if val > 0:
                return "background-color: rgba(38, 166, 154, 0.2); color: #00695c; font-weight: 500;"
            if val < 0:
                return "background-color: rgba(239, 83, 80, 0.2); color: #c62828; font-weight: 500;"
            return ""

        def _cell_style_vs(val):
            if val == "↑ B higher":
                return "background-color: rgba(38, 166, 154, 0.2); color: #00695c; font-weight: 500;"
            if val == "↓ B lower":
                return "background-color: rgba(239, 83, 80, 0.2); color: #c62828; font-weight: 500;"
            return ""

        def _cell_style_easier(val):
            if val == "B":
                return "background-color: rgba(38, 166, 154, 0.25); color: #00695c; font-weight: 600;"
            if val == "A":
                return "background-color: rgba(239, 83, 80, 0.25); color: #c62828; font-weight: 600;"
            if val == "Same":
                return "background-color: rgba(158, 158, 158, 0.15); color: #424242;"
            return ""

        format_cols = {f"{ds_a}": "{:.4g}", f"{ds_b}": "{:.4g}", "diff (B − A)": "{:+.4g}"}
        subset_style = [["diff (B − A)"], ["B vs A"]]
        style_fns = [
            lambda s: [_cell_style_diff(v) for v in s],
            lambda s: [_cell_style_vs(v) for v in s],
        ]
        if is_complexity_summary and "Easier" in df_compare.columns:
            subset_style.append(["Easier"])
            style_fns.append(lambda s: [_cell_style_easier(v) for v in s])

        styled = df_compare.style
        for sub, fn in zip(subset_style, style_fns):
            styled = styled.apply(lambda s, f=fn: f(s), subset=sub)
        styled = styled.format(format_cols, na_rep="—")

        st.markdown(f"**Comparison for group `{group}`**")
        if is_complexity_summary:
            st.caption(
                "**Easier** = less complex dataset (green = B easier, red = A easier). "
                "Hardness (l2, l3, n1–n4, f1, f2, f4): lower is easier. "
                "Ease (l1, f3, landmarking): higher is easier."
            )
        else:
            st.caption("Green tint: B > A · Red tint: B < A")
        st.dataframe(styled, use_container_width=True, hide_index=True)

