"""
Quick launcher for Streamlit app
This script will install Streamlit if needed and launch the application
"""

import subprocess
import sys
import os

def check_streamlit_installed():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit using pip"""
    print("Streamlit not found. Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing Streamlit. Please install manually: pip install streamlit")
        return False

def main():
    """Launch Streamlit app"""
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        if not install_streamlit():
            print("\nPlease install Streamlit manually:")
            print("  pip install streamlit")
            print("\nOr install all requirements:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_script = os.path.join(script_dir, "streamlit_app.py")
    
    # Check if streamlit_app.py exists
    if not os.path.exists(streamlit_script):
        print(f"❌ Error: {streamlit_script} not found!")
        sys.exit(1)
    
    print("🚀 Launching Streamlit application...")
    print(f"📁 Script: {streamlit_script}")
    print("\nThe application will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Run streamlit using python -m streamlit (works even if streamlit command is not in PATH)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_script])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\nTry running manually:")
        print(f"  python -m streamlit run {streamlit_script}")
        sys.exit(1)

if __name__ == "__main__":
    main()



