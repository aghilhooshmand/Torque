"""
Torque DSL - Home Page

Main entry point with navigation to different pages.
"""

import streamlit as st

st.set_page_config(
    page_title="Torque DSL - Home",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Torque DSL")
st.markdown("**Domain-Specific Language for Machine Learning Ensembles**")

st.markdown("---")

# Navigation
st.header("📚 Navigation")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.subheader("🗺️ Mapper")
    st.markdown("""
    - Convert Torque DSL to AST
    - See generated Python / sklearn code
    """)
    if st.button("Go to Mapper", type="primary", use_container_width=True):
        st.switch_page("pages/0_mapper.py")

with col2:
    st.subheader("🏃 Runner")
    st.markdown("""
    - Upload / fetch / create datasets
    - Run Torque DSL commands on data
    - View detailed ML metrics
    """)
    if st.button("Go to Runner", type="primary", use_container_width=True):
        st.switch_page("pages/2_test.py")

with col3:
    st.subheader("📐 Grammar")
    st.markdown("""
    - View and edit grammar file
    - Generate random Torque commands
    - Visualize AST structure
    """)
    if st.button("Go to Grammar", type="primary", use_container_width=True):
        st.switch_page("pages/1_grammar.py")

with col4:
    st.subheader("🧬 Evolution (GE)")
    st.markdown("""
    - Set GE parameters and number of runs
    - Run grammatical evolution for Torque
    - View stats table and train/test chart (mean ± STD)
    """)
    if st.button("Go to Evolution", type="primary", use_container_width=True):
        st.switch_page("pages/3_evolution.py")

with col5:
    st.subheader("📊 Meta-Features")
    st.markdown("""
    - Compute meta-features via PyMFE
    - Store multiple datasets
    - Compare meta-features across datasets
    """)
    if st.button("Go to Meta-Features", type="primary", use_container_width=True):
        st.switch_page("pages/4_meta_features.py")

with col6:
    st.subheader("📖 Guide")
    st.markdown("""
    - Meta-features guide
    - Interpretation help
    - Complexity assessment
    """)
    if st.button("Go to Guide", type="primary", use_container_width=True):
        st.switch_page("pages/5_guide.py")

st.markdown("---")

# About section
st.header("ℹ️ About Torque DSL")

st.markdown("""
Torque DSL is a domain-specific language for creating and testing machine learning ensembles.

**Features:**
- 🎯 Simple syntax for ensemble creation
- 🔧 Support for multiple ensemble types (vote, stack, bag, ada)
- 📊 Classical ML models (LR, DT, NB — fast models for evolution)
- 🌳 Visual AST tree representation
- 📈 Performance metrics evaluation

**Example DSL:**
```python
vote(LR(C=1.0), DT(max_depth=5); voting="hard")
```
""")

st.markdown("---")

# Quick start
st.header("🚀 Quick Start")

with st.expander("How to use Torque DSL"):
    st.markdown("""
    1. **Grammar Page**: Learn the grammar and generate DSL strings
    2. **Test Page**: Upload data (or create mock), then test DSL programs
    3. **Evolution Page**: Use the same data to run Grammatical Evolution; set GE parameters, number of runs, and see per-generation stats and train/test chart (mean ± STD)
    
    **DSL Syntax:**
    - Models: `LR(C=1.0)`, `DT(max_depth=5)`, `NB()`
    - Ensembles: `vote(model1, model2; voting="hard")`
    - Parameters: Use `;` to separate ensemble options from base models
    """)
