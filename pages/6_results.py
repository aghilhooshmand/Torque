"""
Torque DSL - Results Browser

Browse and preview previous experiment results (CSV, HTML, images, JSON, etc.)
from the `results/` folder.
"""

import os
import sys
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image

try:
    import streamlit.components.v1 as components
except Exception:  # pragma: no cover
    components = None

# Project root
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


st.set_page_config(
    page_title="Torque DSL - Results",
    page_icon="📂",
    layout="wide",
)

st.title("📂 Results Browser")
st.markdown("Browse previous experiment outputs (CLI + GUI) under the `results/` folder.")

results_root = os.path.join(current_dir, "results")
if not os.path.isdir(results_root):
    st.error(f"`results/` directory not found under project root: {results_root}")
    st.stop()


def _list_experiment_dirs(root: str) -> List[str]:
    """Return relative paths (from root) of directories that contain at least one file."""
    rel_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if filenames:
            rel = os.path.relpath(dirpath, root)
            rel_paths.append(rel)
    rel_paths = sorted(set(rel_paths))
    return rel_paths


exp_paths = _list_experiment_dirs(results_root)

if not exp_paths:
    st.info("No result files found yet in `results/`. Run Runner/Evolution first.")
    st.stop()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📁 Select experiment folder")
    chosen_rel = st.selectbox(
        "Experiment / results folder",
        options=exp_paths,
        format_func=lambda p: p if p != "." else "(results root)",
    )

    chosen_dir = os.path.join(results_root, chosen_rel)
    st.markdown(f"`{os.path.relpath(chosen_dir, current_dir)}`")

    # List files in chosen directory (non-recursive)
    files = sorted(
        f for f in os.listdir(chosen_dir) if os.path.isfile(os.path.join(chosen_dir, f))
    )

    if not files:
        st.info("This folder has no files (only subfolders). Pick a deeper folder above.")
        selected_file = None
    else:
        st.subheader("📄 Files")
        selected_file = st.radio(
            "Choose a file to preview",
            options=files,
            index=0,
        )

with col_right:
    st.subheader("🔍 Preview")

    if not selected_file:
        st.info("Select a file on the left to see a preview.")
    else:
        path = os.path.join(chosen_dir, selected_file)
        rel_path = os.path.relpath(path, current_dir)
        ext = os.path.splitext(selected_file.lower())[1]

        st.markdown(f"**File:** `{rel_path}`")

        # CSV preview
        if ext == ".csv":
            try:
                df = pd.read_csv(path)
                st.markdown("**CSV preview (first 200 rows):**")
                st.dataframe(df.head(200), use_container_width=True)
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        # HTML preview
        elif ext in {".html", ".htm"}:
            st.markdown("**HTML file:**")
            st.markdown(
                "For full interactivity, open in a browser as well. "
                "Embedded view below (if supported):"
            )
            if components is not None:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    # Larger embedded view
                    components.html(html_content, height=900, scrolling=True)
                    # Button to open in external browser (new tab, if allowed by browser)
                    file_url = "file://" + path.replace(" ", "%20")
                    button_html = f"""
                        <div style="margin-top: 8px;">
                          <button onclick="window.open('{file_url}', '_blank')" style="padding:6px 10px; border-radius:4px; border:1px solid #999; cursor:pointer;">
                            🔗 Open in external browser
                          </button>
                        </div>
                    """
                    components.html(button_html, height=60)
                except Exception as e:
                    st.error(f"Could not embed HTML: {e}")
            else:
                st.info("Streamlit components are not available; cannot embed HTML.")

        # Image preview
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            try:
                img = Image.open(path)
                st.image(img, caption=selected_file, use_column_width=True)
            except Exception as e:
                st.error(f"Could not open image: {e}")

        # JSON preview
        elif ext == ".json":
            try:
                import json

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.json(data)
            except Exception as e:
                st.error(f"Could not read JSON: {e}")

        # PDF or other binary: show a link
        elif ext == ".pdf":
            st.markdown(
                f"PDF file. Open directly from disk:\n\n`{rel_path}`"
            )
        else:
            st.info(
                "Unsupported preview type in the app. "
                "You can open this file directly from disk:\n\n"
                f"`{rel_path}`"
            )

