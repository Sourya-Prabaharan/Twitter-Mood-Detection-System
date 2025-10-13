"""
Example usage script for Twitter Mood Detection System
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from twitter_collector import TwitterCollector
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from analyzer import MoodAnalyzer

def main():
    """Example usage of the Twitter Mood Detection System"""
    
    print("🐦 Twitter Mood Detection System - Example Usage")
    print("=" * 50)
    
    # Check configuration
    try:
        Config.validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set your TWITTER_BEARER_TOKEN in the .env file")
        return
    
    # Initialize components
    print("\n📦 Initializing components...")
    collector = TwitterCollector()
    preprocessor = DataPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    mood_analyzer = MoodAnalyzer()
    
    # Example keywords for mood analysis
    keywords = ["happy", "sad", "excited", "angry", "love", "hate", "amazing", "terrible"]
    
    print(f"\n🔍 Collecting tweets for keywords: {keywords}")
    
    try:
        # Collect tweets
        tweets_df = collector.collect_mood_tweets(keywords, max_tweets_per_keyword=50)
        
        if tweets_df.empty:
            print("❌ No tweets collected. Check your API credentials or try different keywords.")
            return
            
        print(f"✅ Collected {len(tweets_df)} tweets")
        
        # Preprocess tweets
        print("\n🧹 Preprocessing tweets...")
        tweets_df = preprocessor.preprocess_tweets(tweets_df)
        print(f"✅ Preprocessed {len(tweets_df)} tweets")
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"example_tweets_{timestamp}.csv"
        collector.save_tweets_to_csv(tweets_df, filename)
        print(f"💾 Saved tweets to {filename}")
        
        # Analyze sentiment using different methods
        print("\n🧠 Analyzing sentiment...")
        
        # VADER analysis
        print("  - VADER sentiment analysis...")
        vader_results = sentiment_analyzer.analyze_sentiment_batch(
            tweets_df['lemmatized_text'].tolist(),
            method='vader'
        )
        
        # Merge VADER results
        for col in vader_results.columns:
            if col != 'text':
                tweets_df[col] = vader_results[col]
        
        # TextBlob analysis
        print("  - TextBlob sentiment analysis...")
        textblob_results = sentiment_analyzer.analyze_sentiment_batch(
            tweets_df['lemmatized_text'].tolist(),
            method='textblob'
        )
        
        # Merge TextBlob results
        for col in textblob_results.columns:
            if col != 'text':
                tweets_df[col] = textblob_results[col]
        
        # Transformer analysis (if available)
        print("  - Transformer sentiment analysis...")
        transformer_results = sentiment_analyzer.analyze_sentiment_batch(
            tweets_df['lemmatized_text'].tolist(),
            method='transformer'
        )
        
        # Merge Transformer results
        for col in transformer_results.columns:
            if col != 'text':
                tweets_df[col] = transformer_results[col]
        
        print("✅ Sentiment analysis complete!")
        
        # Save results with sentiment analysis
        results_filename = f"sentiment_results_{timestamp}.csv"
        collector.save_tweets_to_csv(tweets_df, results_filename)
        print(f"💾 Saved results to {results_filename}")
        
        # Generate analysis report
        print("\n📊 Generating analysis report...")
        
        # VADER sentiment distribution
        vader_dist = mood_analyzer.calculate_sentiment_distribution(tweets_df, 'vader_sentiment')
        print("\n📈 VADER Sentiment Distribution:")
        print(vader_dist)
        
        # TextBlob sentiment distribution
        textblob_dist = mood_analyzer.calculate_sentiment_distribution(tweets_df, 'textblob_sentiment')
        print("\n📈 TextBlob Sentiment Distribution:")
        print(textblob_dist)
        
        # Overall metrics
        vader_metrics = mood_analyzer.calculate_sentiment_metrics(tweets_df, 'vader_sentiment')
        print("\n📊 VADER Metrics:")
        print(f"  - Total tweets: {vader_metrics['total_tweets']}")
        print(f"  - Dominant sentiment: {vader_metrics['dominant_sentiment']}")
        print(f"  - Sentiment percentages: {vader_metrics['sentiment_percentages']}")
        
        # Keyword analysis
        if 'keyword' in tweets_df.columns:
            print("\n🔍 Sentiment by Keyword:")
            keyword_sentiment = tweets_df.groupby('keyword')['vader_sentiment'].value_counts()
            print(keyword_sentiment)
        
        # Word frequency analysis
        print("\n📝 Most frequent words:")
        word_freq = preprocessor.get_word_frequency(tweets_df)
        print(word_freq.head(20))
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved in: {Config.DATA_DIR}/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


