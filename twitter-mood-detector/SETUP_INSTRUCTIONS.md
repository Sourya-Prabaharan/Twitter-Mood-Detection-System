# 🚀 Quick Setup Instructions

## 1. Environment Setup

```bash
# Navigate to project directory
cd twitter-mood-detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

## 2. Twitter API Configuration

1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Create a developer account and project
3. Generate a Bearer Token
4. Copy `env_example.txt` to `.env`
5. Add your Bearer Token:
   ```
   TWITTER_BEARER_TOKEN=your_actual_bearer_token_here
   ```

## 3. Test the System

```bash
# Run tests
cd src
python test_system.py

# Or run example
python example_usage.py
```

## 4. Launch Dashboard

```bash
# From project root
python run_dashboard.py

# Open browser to http://localhost:8501
```

## 5. Experiment with Jupyter

```bash
# Start Jupyter
jupyter notebook notebooks/experiment.ipynb
```

## 🎯 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Twitter API Bearer Token obtained
- [ ] `.env` file created with Bearer Token
- [ ] System tested (`python src/test_system.py`)
- [ ] Dashboard launched (`python run_dashboard.py`)

## 🔧 Troubleshooting

### "No tweets collected"
- Check Bearer Token is correct
- Verify Twitter API access level
- Try different keywords
- Check rate limits

### "Module not found"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python path

### "Configuration error"
- Verify `.env` file exists
- Check Bearer Token format
- Ensure no extra spaces in `.env` file

## 📞 Need Help?

1. Check the full README.md
2. Run the test suite
3. Review error messages
4. Check Twitter API documentation


