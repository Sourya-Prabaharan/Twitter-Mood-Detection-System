"""
Twitter data collection module
"""
import tweepy
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os

from config import Config

class TwitterCollector:
    """Class for collecting tweets from Twitter API"""
    
    def __init__(self):
        """Initialize Twitter API client"""
        Config.validate_config()
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=Config.TWITTER_BEARER_TOKEN,
            wait_on_rate_limit=True
        )
        
    def search_tweets(self, 
                     query: str, 
                     max_results: int = 100,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Search for tweets using Twitter API v2
        
        Args:
            query: Search query string
            max_results: Maximum number of tweets to retrieve
            start_time: Start time for search (optional)
            end_time: End time for search (optional)
            
        Returns:
            List of tweet dictionaries
        """
        tweets_data = []
        
        try:
            # Convert datetime objects to ISO format strings
            start_time_str = start_time.isoformat() if start_time else None
            end_time_str = end_time.isoformat() if end_time else None
            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(100, max_results),  # API limit is 100 per request
                start_time=start_time_str,
                end_time=end_time_str,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'lang'],
                user_fields=['username', 'location'],
                expansions=['author_id']
            ).flatten(limit=max_results)
            
            # Process tweets
            for tweet in tweets:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'lang': tweet.lang,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count']
                }
                tweets_data.append(tweet_data)
                
        except Exception as e:
            print(f"Error collecting tweets: {e}")
            
        return tweets_data
    
    def collect_mood_tweets(self, 
                           keywords: List[str], 
                           max_tweets_per_keyword: int = 200) -> pd.DataFrame:
        """
        Collect tweets for mood analysis
        
        Args:
            keywords: List of keywords to search for
            max_tweets_per_keyword: Maximum tweets per keyword
            
        Returns:
            DataFrame with collected tweets
        """
        all_tweets = []
        
        for keyword in keywords:
            print(f"Collecting tweets for keyword: {keyword}")
            
            # Create search query
            query = f"{keyword} -is:retweet lang:{Config.TWEET_LANGUAGE}"
            
            # Collect tweets
            tweets = self.search_tweets(
                query=query,
                max_results=max_tweets_per_keyword
            )
            
            # Add keyword label
            for tweet in tweets:
                tweet['keyword'] = keyword
                
            all_tweets.extend(tweets)
            
            # Rate limiting - wait between requests
            time.sleep(1)
            
        return pd.DataFrame(all_tweets)
    
    def save_tweets_to_csv(self, tweets_df: pd.DataFrame, filename: str):
        """
        Save tweets DataFrame to CSV file
        
        Args:
            tweets_df: DataFrame containing tweets
            filename: Output filename
        """
        # Ensure data directory exists
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        
        filepath = os.path.join(Config.DATA_DIR, filename)
        tweets_df.to_csv(filepath, index=False)
        print(f"Tweets saved to {filepath}")
        
    def load_tweets_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load tweets from CSV file
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame containing tweets
        """
        filepath = os.path.join(Config.DATA_DIR, filename)
        return pd.read_csv(filepath)
