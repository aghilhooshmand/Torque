"""
Torque DSL - Mapper Page

Map a Torque DSL command to:
- AST (as JSON)
- Generated Python code for a scikit-learn estimator
"""

import os
import sys
import json

import streamlit as st
from html import escape as html_escape

# Project root
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Torque_mapper import TorqueMapper  # noqa: E402


def _build_ast_tree_html(node: dict, level: int = 0, is_last: bool = True, prefix: str = "") -> str:
    """Build HTML representation of AST tree (same style as Grammar page)."""
    if node.get("type") == "call":
        name = node.get("name", "?")
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
        html += "</div>"

        pos_args = node.get("pos", [])
        for i, pos_arg in enumerate(pos_args):
            is_last_pos = i == len(pos_args) - 1 and not node.get("kw")
            html += _build_ast_tree_html(pos_arg, level + 1, is_last_pos, new_prefix)

        kw_args = node.get("kw", {})
        for i, (key, value) in enumerate(kw_args.items()):
            is_last_kw = i == len(kw_args) - 1
            indent = (level + 1) * 20
            kw_connector = "└── " if is_last_kw else "├── "

            if isinstance(value, dict) and value.get("type") == "literal":
                value_str = str(value.get("value", ""))
                if len(value_str) > 30:
                    value_str = value_str[:27] + "..."
                if isinstance(value.get("value"), str):
                    value_display = f'"{html_escape(value_str)}"'
                else:
                    value_display = html_escape(value_str)
            else:
                value_display = f"{value.get('name', '?')}(...)" if isinstance(value, dict) else html_escape(str(value))

            html += f'<div style="font-family: monospace; margin-left: {indent}px; line-height: 1.6;">'
            html += f'<span style="color: #009900;">{html_escape(kw_connector)}{html_escape(str(key))}=</span>'
            html += f'<span style="color: #cc6600;"> {value_display}</span>'
            html += "</div>"

        return html

    if node.get("type") == "literal":
        value = node.get("value")
        value_str = str(value)
        if len(value_str) > 30:
            value_str = value_str[:27] + "..."
        indent = level * 20
        html = f'<div style="font-family: monospace; margin-left: {indent}px; line-height: 1.6;">'
        html += f'<span style="color: #cc6600;">└── `{html_escape(value_str)}`</span>'
        html += "</div>"
        return html

    return ""


st.set_page_config(
    page_title="Torque DSL - Mapper",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Torque Mapper")
st.caption("Convert a Torque DSL command into its AST and generated Python code.")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("✍️ Torque Command")

    example = 'vote(LR(C=1.0), DT(max_depth=5); voting="hard")'

    dsl_string = st.text_area(
        "Enter Torque DSL command",
        value=st.session_state.get("mapper_dsl", example),
        height=150,
        key="mapper_dsl_editor",
        help="Write a Torque command to map into AST and Python code.",
    )

    st.markdown("**Example:**")
    st.code(example, language="python")

    run_mapping = st.button("🧭 Map to AST & Python", type="primary", use_container_width=True)

with col_right:
    st.subheader("🧩 Mapping Result")

    if run_mapping:
        if not dsl_string.strip():
            st.error("❌ Please enter a Torque DSL command.")
        else:
            try:
                mapper = TorqueMapper()

                # AST
                ast = mapper.dsl_to_ast(dsl_string)

                # Generated Python code
                python_code = mapper.map_to_python(dsl_string, variable_name="estimator")

                st.success("✅ Mapping successful.")

                with st.expander("🌳 AST (JSON)", expanded=False):
                    st.json(ast)

                with st.expander("🌳 AST Tree View", expanded=True):
                    try:
                        tree_html = _build_ast_tree_html(ast)
                        st.markdown(tree_html, unsafe_allow_html=True)
                    except Exception as tree_err:
                        st.warning(f"Could not render AST tree: {tree_err}")

                with st.expander("🐍 Generated Python Code", expanded=True):
                    st.code(python_code, language="python")

                # Optionally allow saving mapping as JSON
                with st.expander("💾 Export mapping to JSON"):
                    default_filename = os.path.join("results", "Torque_mapper_result.json")
                    filename = st.text_input(
                        "Output JSON file path",
                        value=default_filename,
                        help="Path relative to project root. Results go in results/.",
                    )
                    if st.button("Save mapping JSON", use_container_width=True):
                        try:
                            output_path = os.path.join(current_dir, filename)
                            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                            payload = {
                                "torque_command": dsl_string,
                                "ast": ast,
                                "python_code": python_code,
                                "variable_name": "estimator",
                            }
                            with open(output_path, "w") as f:
                                json.dump(payload, f, indent=2)
                            st.success(f"✅ Saved mapping to `{output_path}`")
                        except Exception as e:
                            st.error(f"❌ Could not save mapping JSON: {e}")

            except Exception as e:
                st.error(f"❌ Error while mapping Torque DSL: {e}")

    else:
        st.info("💡 Enter a Torque DSL command on the left, then click **Map to AST & Python**.")

