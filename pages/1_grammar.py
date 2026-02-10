"""
Torque DSL - Grammar Page

Grammar definition and DSL generation.
"""

import sys
import os

import streamlit as st
from html import escape as html_escape

# Import modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Torque_mapper import map_dsl_to_ast as parse_dsl_to_ast
from grammar import GRAMMAR, GRAMMAR_BNF, generate_dsl_from_grammar

GRAMMAR_FILE = os.path.join(current_dir, "grammar", "ensamble_grammar.bnf")

st.set_page_config(
    page_title="Torque DSL - Grammar",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Grammar & DSL Generation")


def _read_grammar_file() -> str:
    """Read the BNF grammar file from disk."""
    try:
        with open(GRAMMAR_FILE, "r") as f:
            return f.read()
    except Exception:
        # Fallback to in-memory BNF from grammar module
        return GRAMMAR_BNF


if "grammar_text" not in st.session_state:
    st.session_state.grammar_text = _read_grammar_file()
if "grammar_edit_mode" not in st.session_state:
    st.session_state.grammar_edit_mode = False

# Define tree function at top
def _build_ast_tree_html(node: dict, level: int = 0, is_last: bool = True, prefix: str = "") -> str:
    """Build HTML representation of AST tree."""
    if node["type"] == "call":
        name = node["name"]
        pos_count = len(node.get("pos", []))
        kw_count = len(node.get("kw", {}))
        
        if level == 0:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
        
        html = f'<div style="font-family: monospace; margin-left: {level * 20}px; line-height: 1.6;">'
        html += f'<span style="color: #0066cc; font-weight: bold;">{html_escape(connector)}{html_escape(name)}</span>'
        html += f' <span style="color: #666; font-size: 0.9em;">(pos: {pos_count}, kw: {kw_count})</span>'
        html += '</div>'
        
        pos_args = node.get("pos", [])
        for i, pos_arg in enumerate(pos_args):
            is_last_pos = i == len(pos_args) - 1 and not node.get("kw")
            html += _build_ast_tree_html(pos_arg, level + 1, is_last_pos, new_prefix)
        
        kw_args = node.get("kw", {})
        for i, (key, value) in enumerate(kw_args.items()):
            is_last_kw = i == len(kw_args) - 1
            indent = (level + 1) * 20
            kw_connector = "└── " if is_last_kw else "├── "
            
            if value.get("type") == "literal":
                value_str = str(value.get("value", ""))
                if len(value_str) > 30:
                    value_str = value_str[:27] + "..."
                if isinstance(value.get("value"), str):
                    value_display = f'"{html_escape(value_str)}"'
                else:
                    value_display = html_escape(value_str)
            else:
                value_display = f"{value.get('name', '?')}(...)"
            
            html += f'<div style="font-family: monospace; margin-left: {indent}px; line-height: 1.6;">'
            html += f'<span style="color: #009900;">{html_escape(kw_connector)}{html_escape(key)}=</span>'
            html += f'<span style="color: #cc6600;"> {value_display}</span>'
            html += '</div>'
        
        return html
    
    elif node["type"] == "literal":
        value = node["value"]
        value_str = str(value)
        if len(value_str) > 30:
            value_str = value_str[:27] + "..."
        indent = level * 20
        html = f'<div style="font-family: monospace; margin-left: {indent}px; line-height: 1.6;">'
        html += f'<span style="color: #cc6600;">└── `{html_escape(value_str)}`</span>'
        html += '</div>'
        return html
    
    return ""

# ============================================
# PART 1: Grammar Definition
# ============================================
st.header("Part 1: Grammar Definition")
st.markdown("**Simple grammar for classical ML ensemble learning**")

col_grammar_left, col_grammar_right = st.columns([2, 1])

with col_grammar_left:
    st.subheader("📐 Grammar (BNF file)")
    st.markdown("File: `grammar/ensamble_grammar.bnf`")

    if not st.session_state.grammar_edit_mode:
        # Read-only, syntax-highlighted view
        st.code(st.session_state.grammar_text, language="bnf")
        if st.button("✏️ Edit Grammar", use_container_width=True):
            st.session_state.grammar_edit_mode = True
            st.rerun()
    else:
        # Editable text area
        grammar_text = st.text_area(
            "Edit Grammar (BNF)",
            value=st.session_state.grammar_text,
            height=300,
            key="grammar_editor",
        )
        st.session_state.grammar_text = grammar_text

        col_btn_save, col_btn_cancel = st.columns(2)
        with col_btn_save:
            if st.button("💾 Save Grammar to File", use_container_width=True):
                try:
                    with open(GRAMMAR_FILE, "w") as f:
                        f.write(st.session_state.grammar_text)
                    st.session_state.grammar_edit_mode = False
                    st.success(f"✅ Saved grammar to `{GRAMMAR_FILE}`. Reload app to use updated grammar.")
                except Exception as e:
                    st.error(f"❌ Could not save grammar: {e}")
        with col_btn_cancel:
            if st.button("↩️ Cancel", use_container_width=True):
                # Reload from file and exit edit mode
                st.session_state.grammar_text = _read_grammar_file()
                st.session_state.grammar_edit_mode = False
                st.rerun()

with col_grammar_right:
    st.subheader("📊 Grammar Info")
    
    total_rules = len(GRAMMAR)
    total_productions = sum(len(prods) for prods in GRAMMAR.values())
    
    st.metric("Total Rules", total_rules)
    st.metric("Total Productions", total_productions)
    
    st.markdown("**Available Models:**")
    models = ["LR", "DT", "NB"]
    for model in models:
        st.write(f"- {model}")
    
    st.markdown("**Ensemble Types:**")
    st.write("- vote (VotingClassifier)")
    st.write("- stack (StackingClassifier)")
    st.write("- bag (BaggingClassifier)")
    st.write("- ada (AdaBoostClassifier)")

st.markdown("---")

# ============================================
# PART 2: Generate Random DSL String
# ============================================
st.header("Part 2: Generate Random DSL String")

col_gen_left, col_gen_right = st.columns([1, 2])

with col_gen_left:
    st.subheader("⚙️ Generation Settings")
    
    max_depth = st.slider(
        "Max Depth",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
        help="Maximum recursion depth for grammar expansion"
    )
    
    if st.button("🎲 Generate Random DSL", type="primary", use_container_width=True):
        try:
            generated_dsl = generate_dsl_from_grammar(max_depth=max_depth)
            st.session_state.generated_dsl = generated_dsl
            st.session_state.generation_success = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Generation error: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.session_state.generation_success = False

with col_gen_right:
    st.subheader("📝 Generated DSL String")
    
    if "generated_dsl" in st.session_state and st.session_state.get("generation_success", False):
        generated = st.session_state.generated_dsl
        
        st.code(generated, language="python")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("DSL Length", len(generated))
        with col_info2:
            model_count = sum(1 for m in ["LR", "DT", "NB"] if m in generated)
            st.metric("Models Found", model_count)
        
        if st.button("📋 Copy DSL String", use_container_width=True):
            st.code(generated, language="python")
            st.success("✅ DSL string copied! (Use Ctrl+C to copy from the code block above)")
        
        st.markdown("---")
        st.subheader("🌳 AST Tree View")
        
        try:
            ast = parse_dsl_to_ast(generated)
            st.session_state.generated_ast = ast
            
            tree_html = _build_ast_tree_html(ast)
            st.markdown(tree_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Could not parse DSL: {e}")
            st.info("The generated DSL might not be valid. Try generating again.")
        
    else:
        st.info("💡 Click 'Generate Random DSL' to create a DSL string from the grammar")
        
        st.markdown("**Example output:**")
        example = 'vote(LR(C=1.0), DT(max_depth=5); voting="hard")'
        st.code(example, language="python")
        
        st.markdown("---")
        st.subheader("🌳 Example AST Tree View")
        try:
            example_ast = parse_dsl_to_ast(example)
            example_tree_html = _build_ast_tree_html(example_ast)
            st.markdown(example_tree_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not parse example: {e}")

