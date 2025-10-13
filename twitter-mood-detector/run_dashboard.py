"""
Main script to run the Twitter Mood Detection Dashboard
"""
import os
import sys
import subprocess

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Twitter Mood Detection Dashboard...")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        print("⚠️  Warning: .env file not found!")
        print("📝 Please create a .env file with your Twitter API credentials.")
        print("   You can use env_example.txt as a template.")
        
        # Check if env_example.txt exists
        env_example = os.path.join(os.path.dirname(__file__), 'env_example.txt')
        if os.path.exists(env_example):
            print(f"   Template available at: {env_example}")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("👋 Goodbye!")
            return
    
    # Run Streamlit dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard.py')
    
    try:
        print(f"🌐 Starting dashboard at: {dashboard_path}")
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            dashboard_path, '--server.port', '8501'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify your .env file has the correct Twitter API credentials")

if __name__ == "__main__":
    main()


