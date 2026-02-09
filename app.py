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

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Grammar")
    st.markdown("""
    - View grammar rules for DSL generation
    - Generate random DSL strings from grammar
    - See AST tree visualization
    """)
    if st.button("Go to Grammar Page", type="primary", use_container_width=True):
        st.switch_page("pages/1_grammar.py")

with col2:
    st.subheader("🧪 Test DSL")
    st.markdown("""
    - Upload or create datasets
    - Write and test DSL programs
    - View performance metrics
    """)
    if st.button("Go to Test Page", type="primary", use_container_width=True):
        st.switch_page("pages/2_test.py")

with col3:
    st.subheader("🧬 Evolution")
    st.markdown("""
    - Set GE parameters and number of runs
    - Run grammatical evolution for Torque
    - View stats table and train/test chart (mean ± STD)
    """)
    if st.button("Go to Evolution Page", type="primary", use_container_width=True):
        st.switch_page("pages/3_evolution.py")

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
