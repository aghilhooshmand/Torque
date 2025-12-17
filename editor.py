"""
Simple text editor for writing Torque DSL programs.

This is a basic editor with text area and common editor commands.
"""

import re
from typing import Any

import streamlit as st


def main():
    st.set_page_config(
        page_title="Torque DSL Editor",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Torque DSL Editor")
    
    # Initialize session state for editor content
    if "editor_content" not in st.session_state:
        st.session_state.editor_content = ""
    
    if "clipboard" not in st.session_state:
        st.session_state.clipboard = ""
    
    if "selected_function_category" not in st.session_state:
        st.session_state.selected_function_category = "ML Models"
    
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = None
    
    # Command buttons row - equal size and spacing
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        if st.button("📋 Copy", use_container_width=True):
            if st.session_state.editor_content:
                st.session_state.clipboard = st.session_state.editor_content
                st.success("Text copied to clipboard!")
            else:
                st.warning("No text to copy")
    
    with col2:
        if st.button("✂️ Cut", use_container_width=True):
            if st.session_state.editor_content:
                st.session_state.clipboard = st.session_state.editor_content
                st.session_state.editor_content = ""
                st.success("Text cut to clipboard!")
                st.rerun()
            else:
                st.warning("No text to cut")
    
    with col3:
        if st.button("📄 Paste", use_container_width=True):
            if st.session_state.clipboard:
                current_text = st.session_state.editor_content
                st.session_state.editor_content = current_text + st.session_state.clipboard
                st.success("Text pasted!")
                st.rerun()
            else:
                st.warning("Clipboard is empty")
    
    with col4:
        if st.button("🗑️ Delete", use_container_width=True):
            if st.session_state.editor_content:
                st.session_state.editor_content = ""
                st.success("Text deleted!")
                st.rerun()
            else:
                st.warning("No text to delete")
    
    with col5:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.editor_content = ""
            st.session_state.clipboard = ""
            st.success("Editor cleared!")
            st.rerun()
    
    with col6:
        if st.button("💾 Save", use_container_width=True):
            st.success("Content saved in session!")
    
    st.markdown("---")
    
    # Function selector section
    with st.expander("🔧 Function Builder (Click to insert functions)", expanded=False):
        col_cat, col_func = st.columns(2)
        
        with col_cat:
            category = st.selectbox(
                "Function Category",
                ["ML Models", "Aggregate Functions"],
                key="func_category",
                index=0 if st.session_state.selected_function_category == "ML Models" else 1
            )
            st.session_state.selected_function_category = category
        
        with col_func:
            if category == "ML Models":
                function_name = st.selectbox(
                    "ML Model",
                    ["RF", "LR", "SVM", "DT", "XGB"],
                    key="ml_model_select"
                )
                st.session_state.selected_function = function_name
                
                # ML Model parameters based on scikit-learn
                st.markdown("**Parameters:**")
                if function_name == "RF":
                    n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1, key="rf_n_est")
                    max_depth = st.number_input("max_depth (0=unlimited)", min_value=0, value=0, step=1, key="rf_max_d")
                    max_depth_val = None if max_depth == 0 else max_depth
                    min_samples_split = st.number_input("min_samples_split", min_value=2, value=2, step=1, key="rf_min_split")
                    random_state = st.number_input("random_state", min_value=0, value=42, step=1, key="rf_rand")
                    
                    # Build function call
                    params = [f"n_estimators={n_estimators}"]
                    if max_depth_val is not None:
                        params.append(f"max_depth={max_depth_val}")
                    params.append(f"min_samples_split={min_samples_split}")
                    params.append(f"random_state={random_state}")
                    function_call = f"{function_name}({', '.join(params)})"
                
                elif function_name == "LR":
                    C = st.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.1, key="lr_c")
                    penalty = st.selectbox("penalty", ["l2", "l1"], key="lr_penalty")
                    max_iter = st.number_input("max_iter", min_value=1, value=100, step=10, key="lr_max_iter")
                    random_state = st.number_input("random_state", min_value=0, value=42, step=1, key="lr_rand")
                    
                    function_call = f"{function_name}(C={C}, penalty=\"{penalty}\", max_iter={max_iter}, random_state={random_state})"
                
                elif function_name == "SVM":
                    C = st.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.1, key="svm_c")
                    kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
                    gamma = st.selectbox("gamma", ["scale", "auto"], key="svm_gamma")
                    random_state = st.number_input("random_state", min_value=0, value=42, step=1, key="svm_rand")
                    
                    function_call = f"{function_name}(C={C}, kernel=\"{kernel}\", gamma=\"{gamma}\", random_state={random_state})"
                
                elif function_name == "DT":
                    criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], key="dt_criterion")
                    max_depth = st.number_input("max_depth (0=unlimited)", min_value=0, value=0, step=1, key="dt_max_d")
                    max_depth_val = None if max_depth == 0 else max_depth
                    min_samples_split = st.number_input("min_samples_split", min_value=2, value=2, step=1, key="dt_min_split")
                    random_state = st.number_input("random_state", min_value=0, value=42, step=1, key="dt_rand")
                    
                    params = [f"criterion=\"{criterion}\""]
                    if max_depth_val is not None:
                        params.append(f"max_depth={max_depth_val}")
                    params.append(f"min_samples_split={min_samples_split}")
                    params.append(f"random_state={random_state}")
                    function_call = f"{function_name}({', '.join(params)})"
                
                else:  # XGB
                    n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1, key="xgb_n_est")
                    learning_rate = st.number_input("learning_rate", min_value=0.01, value=0.1, step=0.01, key="xgb_lr")
                    max_depth = st.number_input("max_depth", min_value=1, value=6, step=1, key="xgb_max_d")
                    random_state = st.number_input("random_state", min_value=0, value=42, step=1, key="xgb_rand")
                    
                    function_call = f"{function_name}(n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, random_state={random_state})"
            
            else:  # Aggregate Functions
                function_name = st.selectbox(
                    "Aggregate Function",
                    ["avg", "sum", "max", "min", "count", "mean"],
                    key="agg_func_select"
                )
                st.session_state.selected_function = function_name
                
                st.markdown("**Parameters:**")
                column_name = st.text_input("column (parameter)", value="value", key="agg_col")
                
                if function_name in ["avg", "sum", "max", "min", "mean"]:
                    function_call = f"{function_name}(column=\"{column_name}\")"
                else:  # count
                    function_call = f"{function_name}(column=\"{column_name}\")"
        
        # Preview and Insert button
        st.markdown("---")
        st.markdown(f"**Preview:** `{function_call}`")
        
        col_insert, col_preview = st.columns([1, 3])
        with col_insert:
            if st.button("➕ Insert Function", use_container_width=True, type="primary"):
                # Insert function at the end of current text
                current_text = st.session_state.editor_content
                if current_text.strip():
                    # Add comma if needed
                    if not current_text.rstrip().endswith(','):
                        st.session_state.editor_content = current_text + ", " + function_call
                    else:
                        st.session_state.editor_content = current_text + " " + function_call
                else:
                    st.session_state.editor_content = function_call
                st.success(f"Inserted: {function_call}")
                st.rerun()
    
    st.markdown("---")
    
    # Editor text area - in the middle (3 lines, full width)
    editor_text = st.text_area(
        label="DSL Program Editor",
        value=st.session_state.editor_content,
        height=75,  # Approximately 3 lines
        key="editor_textarea",
        help="Type your DSL program here. Example: func1(param1=value1, func2(param2=123), param3=\"text\")",
        label_visibility="collapsed",
    )
    
    # Update session state with current content
    st.session_state.editor_content = editor_text
    
    st.markdown("---")
    
    # Statistics and Report - at the bottom
    st.subheader("📊 Statistics & Report")
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric("Characters", len(editor_text))
    
    with col_info2:
        st.metric("Lines", editor_text.count('\n') + 1 if editor_text else 0)
    
    with col_info3:
        words = len(editor_text.split()) if editor_text else 0
        st.metric("Words", words)
    
    with col_info4:
        # Count functions in the text
        if editor_text:
            # Simple heuristic: count function calls (identifier followed by '(')
            # This counts all function calls including nested ones
            # Pattern: word character sequence followed by '('
            function_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\('
            function_count = len(re.findall(function_pattern, editor_text))
        else:
            function_count = 0
        st.metric("Functions", function_count)
    
    # Display clipboard content
    if st.session_state.clipboard:
        with st.expander("📋 Clipboard Content"):
            st.code(st.session_state.clipboard, language="text")


if __name__ == "__main__":
    main()

