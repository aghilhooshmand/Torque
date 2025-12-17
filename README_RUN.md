# How to Run the Torque DSL App

## Quick Start

```bash
# Activate virtual environment
source torque-env/bin/activate

# Run the app
streamlit run app.py
```

## If the Interface Doesn't Update

If you see the old interface or changes don't appear:

1. **Stop the Streamlit server** (Ctrl+C in the terminal)

2. **Clear Streamlit cache:**
   ```bash
   rm -rf .streamlit/cache
   ```

3. **Restart with auto-reload:**
   ```bash
   streamlit run app.py --server.runOnSave true
   ```

4. **Or use the helper script:**
   ```bash
   ./run_app.sh
   ```

## Troubleshooting

- **Import errors**: Make sure you're in the project directory and the virtual environment is activated
- **Module not found**: Run `pip install -r requirements.txt` to install dependencies
- **Interface not updating**: Clear browser cache (Ctrl+Shift+R) and Streamlit cache

## What You Should See

The app should show:
1. **Step 0: Grammar** - View grammar rules and generate DSL
2. **Step 1: DSL String** - Editor for writing DSL programs
3. **Step 2: Parser** - Shows AST structure
4. **Step 3: Compiler** - Shows compiled sklearn estimator
5. **Step 4: Evaluator** - Cross-validation evaluation

