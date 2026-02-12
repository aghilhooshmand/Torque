"""
Torque DSL - Meta-Features Guide

Display comprehensive guide for understanding meta-features and dataset complexity.
"""

import os
import streamlit as st

st.set_page_config(
    page_title="Torque DSL - Meta-Features Guide",
    page_icon="📖",
    layout="wide",
)

st.title("📖 Meta-Features Guide")
st.markdown("**Understanding Dataset Complexity**")

# Read the guide markdown file
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
guide_path = os.path.join(current_dir, "Document", "META_FEATURES_GUIDE.md")

try:
    with open(guide_path, "r", encoding="utf-8") as f:
        guide_content = f.read()
    
    # Display the markdown content
    st.markdown(guide_content)
    
except FileNotFoundError:
    st.error(f"❌ Guide file not found at: {guide_path}")
    st.info("Please ensure `Document/META_FEATURES_GUIDE.md` exists.")
except Exception as e:
    st.error(f"❌ Error reading guide file: {e}")

# Footer with navigation
st.markdown("---")
st.markdown("**💡 Tip**: Use this guide while exploring meta-features on the **Meta-Features** page!")

col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("Go to Meta-Features Page →", use_container_width=True, type="primary"):
        st.switch_page("pages/4_meta_features.py")
