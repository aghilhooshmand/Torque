"""Quick test to verify app.py can be imported and main() works."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import app
    print("✅ App imports successfully")
    
    # Check if main function exists
    if hasattr(app, 'main'):
        print("✅ main() function exists")
    else:
        print("❌ main() function not found")
    
    # Check if grammar is accessible
    if hasattr(app, 'GRAMMAR'):
        print(f"✅ Grammar loaded: {len(app.GRAMMAR)} rules")
    else:
        print("❌ Grammar not found")
    
    print("\n✅ App structure looks good!")
    print("\nTo run the app, use:")
    print("  streamlit run app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

