"""
Configuration module for Twitter Mood Detection System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Twitter Mood Detection System"""
    
    # Twitter API Configuration
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    # Data Configuration
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    
    # Tweet Collection Configuration
    MAX_TWEETS_PER_QUERY = 1000
    TWEET_LANGUAGE = 'en'
    
    # Sentiment Analysis Configuration
    SENTIMENT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
    
    # Dashboard Configuration
    DASHBOARD_TITLE = "Twitter Mood Detection System"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.TWITTER_BEARER_TOKEN:
            raise ValueError("TWITTER_BEARER_TOKEN is required. Please set it in your .env file")
        return True
