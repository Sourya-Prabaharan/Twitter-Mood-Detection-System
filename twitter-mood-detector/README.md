# 🐦 Twitter Mood Detection System

A comprehensive system for collecting, analyzing, and visualizing sentiment from Twitter data using multiple machine learning approaches.

## 🌟 Features

- **Real-time Tweet Collection**: Fetch tweets using Twitter API v2
- **Multiple Sentiment Analysis Methods**:
  - VADER Sentiment Analysis (lexicon-based)
  - TextBlob (rule-based)
  - Pre-trained Transformer Models (BERT-based)
  - Custom trained models
- **Advanced Data Preprocessing**: Text cleaning, tokenization, lemmatization
- **Interactive Dashboard**: Beautiful Streamlit web interface
- **Comprehensive Analytics**: Visualizations, word clouds, timeline analysis
- **Export Capabilities**: CSV, JSON report exports

## 📁 Project Structure

```
twitter-mood-detector/
├── src/
│   ├── config.py              # Configuration management
│   ├── twitter_collector.py   # Twitter API integration
│   ├── data_preprocessor.py   # Text preprocessing
│   ├── sentiment_analyzer.py  # Sentiment analysis methods
│   ├── analyzer.py           # Data analysis and visualization
│   ├── dashboard.py          # Streamlit dashboard
│   └── example_usage.py      # Example usage script
├── data/                     # Tweet data storage
├── models/                   # Trained models storage
├── notebooks/               # Jupyter notebooks (optional)
├── requirements.txt         # Python dependencies
├── env_example.txt         # Environment variables template
├── .gitignore             # Git ignore rules
├── run_dashboard.py       # Dashboard launcher
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Twitter Developer Account
- Twitter API Bearer Token

### 2. Setup

1. **Clone or download the project**:
   ```bash
   git clone <your-repo-url>
   cd twitter-mood-detector
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Twitter API credentials**:
   - Copy `env_example.txt` to `.env`
   - Add your Twitter Bearer Token:
     ```
     TWITTER_BEARER_TOKEN=your_bearer_token_here
     ```

### 3. Running the System

#### Option A: Interactive Dashboard (Recommended)
```bash
python run_dashboard.py
```
Then open your browser to `http://localhost:8501`

#### Option B: Command Line Usage
```bash
cd src
python example_usage.py
```

## 🔧 Twitter API Setup

1. **Create Developer Account**:
   - Go to [developer.twitter.com](https://developer.twitter.com)
   - Apply for a developer account
   - Create a new project/app

2. **Generate Bearer Token**:
   - Go to your project's "Keys and Tokens" tab
   - Generate a Bearer Token
   - Copy the token to your `.env` file

3. **API Access Levels**:
   - **Free Tier**: 10,000 tweets per month
   - **Basic Tier**: 10,000 tweets per month with additional features
   - **Pro/Enterprise**: Higher limits and advanced features

## 📊 Usage Guide

### Dashboard Interface

The Streamlit dashboard provides four main tabs:

1. **📈 Overview**: System status and quick statistics
2. **🔍 Data Collection**: Collect tweets with custom keywords
3. **📊 Analysis**: Run sentiment analysis and view visualizations
4. **📝 Reports**: Generate and export comprehensive reports

### Key Features

#### Tweet Collection
- Enter comma-separated keywords
- Set maximum tweets per keyword (50-500)
- Filter by language (default: English)
- Exclude retweets for original content

#### Sentiment Analysis Methods
- **VADER**: Fast, lexicon-based, good for social media
- **TextBlob**: Simple rule-based approach
- **Transformer**: State-of-the-art accuracy with BERT models
- **All Methods**: Compare results across different approaches

#### Visualizations
- Sentiment distribution pie charts
- Timeline analysis showing mood trends
- Keyword-based sentiment breakdown
- Word clouds for positive/negative/neutral tweets
- Method comparison charts

## 🧪 Testing and Validation

### Test with Different Keywords

Try these keyword sets for testing:

```python
# Emotion-based
emotions = ["happy", "sad", "excited", "angry", "love", "hate"]

# Topic-based
topics = ["climate", "politics", "technology", "sports", "music"]

# Brand-based
brands = ["apple", "microsoft", "google", "tesla", "netflix"]

# Event-based
events = ["election", "olympics", "holiday", "graduation", "wedding"]
```

### Accuracy Evaluation

1. **Manual Labeling**: Label a sample of tweets manually
2. **Cross-validation**: Compare different methods on the same dataset
3. **Domain-specific Testing**: Test on different topics and languages

## 🔍 Advanced Features

### Custom Model Training

Train your own sentiment classifier:

```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Train on labeled data
performance = analyzer.train_custom_model(
    df_labeled_data, 
    text_column='text',
    label_column='sentiment'
)

# Use custom model for predictions
predictions = analyzer.predict_with_custom_model(texts)
```

### Batch Processing

Process large datasets efficiently:

```python
from src.twitter_collector import TwitterCollector

collector = TwitterCollector()

# Collect tweets in batches
keywords = ["happy", "sad", "angry"]
tweets_df = collector.collect_mood_tweets(keywords, max_tweets_per_keyword=1000)

# Save and load data
collector.save_tweets_to_csv(tweets_df, "large_dataset.csv")
loaded_df = collector.load_tweets_from_csv("large_dataset.csv")
```

## 📈 Performance Optimization

### Memory Management
- Process tweets in batches for large datasets
- Use data types optimization (e.g., category for sentiment labels)
- Clear intermediate variables when processing large files

### API Rate Limits
- Built-in rate limiting with `wait_on_rate_limit=True`
- Automatic pagination for large tweet collections
- Configurable delays between requests

### Model Optimization
- Use GPU acceleration for transformer models when available
- Cache pre-trained models to avoid reloading
- Batch processing for sentiment analysis

## 🛠️ Troubleshooting

### Common Issues

1. **"No tweets collected"**:
   - Check your Bearer Token is valid
   - Verify keywords are not too specific
   - Try different time ranges or languages

2. **"Transformer model not available"**:
   - Ensure internet connection for model download
   - Check available disk space
   - Verify transformers library installation

3. **"Rate limit exceeded"**:
   - Wait for rate limit reset (usually 15 minutes)
   - Reduce batch sizes
   - Use fewer concurrent requests

4. **Memory issues with large datasets**:
   - Process data in smaller chunks
   - Use data type optimization
   - Consider using Dask or similar for big data

### Getting Help

1. Check the error logs in the terminal
2. Verify your `.env` file configuration
3. Test with a small dataset first
4. Check Twitter API status and your account limits

## 📚 Dependencies

### Core Libraries
- **tweepy**: Twitter API client
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **transformers**: Hugging Face transformer models
- **torch**: PyTorch for deep learning

### Visualization
- **streamlit**: Web dashboard
- **plotly**: Interactive charts
- **matplotlib/seaborn**: Static plots
- **wordcloud**: Word cloud generation

### NLP Libraries
- **nltk**: Natural language processing
- **textblob**: Simple sentiment analysis
- **vaderSentiment**: VADER sentiment analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Twitter API for providing access to tweet data
- Hugging Face for pre-trained transformer models
- The open-source community for excellent Python libraries
- Streamlit team for the amazing dashboard framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the example usage scripts

---

**Happy Mood Detection! 🐦✨**


