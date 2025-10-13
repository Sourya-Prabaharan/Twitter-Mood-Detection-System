"""
Setup script for Twitter Mood Detection System
"""
import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ['data', 'models', 'notebooks']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def setup_env_file():
    """Setup environment file"""
    print("\n🔧 Setting up environment file...")
    
    env_file = '.env'
    env_example = 'env_example.txt'
    
    if os.path.exists(env_file):
        print(f"✅ Environment file already exists: {env_file}")
    else:
        if os.path.exists(env_example):
            # Copy example to .env
            with open(env_example, 'r') as src:
                content = src.read()
            
            with open(env_file, 'w') as dst:
                dst.write(content)
            
            print(f"✅ Created {env_file} from {env_example}")
            print("⚠️  Please edit .env file with your Twitter API credentials")
        else:
            print(f"❌ {env_example} not found")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        'tweepy',
        'pandas',
        'sklearn',
        'nltk',
        'transformers',
        'torch',
        'streamlit',
        'textblob',
        'vaderSentiment',
        'wordcloud',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All modules imported successfully")
        return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def main():
    """Main setup function"""
    print("🐦 Twitter Mood Detection System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Create directories
    create_directories()
    
    # Setup environment file
    setup_env_file()
    
    # Download NLTK data
    download_nltk_data()
    
    # Test imports
    if test_imports():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your Twitter API Bearer Token")
        print("2. Run the dashboard: python run_dashboard.py")
        print("3. Or test with example: cd src && python example_usage.py")
    else:
        print("\n❌ Setup completed with errors")
        print("   Please check the failed imports and try again")

if __name__ == "__main__":
    main()


