"""
Test script for Twitter Mood Detection System
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

def test_configuration():
    """Test configuration setup"""
    print("🔧 Testing configuration...")
    try:
        Config.validate_config()
        print("✅ Configuration valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\n🧹 Testing data preprocessing...")
    try:
        preprocessor = DataPreprocessor()
        
        # Test data
        test_texts = [
            "I love this movie! 😍 #amazing",
            "This is terrible... :(",
            "Neutral text here.",
            "RT @user: This is a retweet",
            "https://example.com check this out"
        ]
        
        # Test cleaning
        for text in test_texts:
            cleaned = preprocessor.clean_text(text)
            no_stopwords = preprocessor.remove_stopwords(cleaned)
            lemmatized = preprocessor.lemmatize_text(no_stopwords)
            print(f"  Original: {text}")
            print(f"  Cleaned: {clemmatized}")
            print()
        
        print("✅ Data preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n🧠 Testing sentiment analysis...")
    try:
        analyzer = SentimentAnalyzer()
        
        # Test texts
        test_texts = [
            "I love this amazing product!",
            "This is absolutely terrible.",
            "The weather is okay today.",
            "I'm so excited about this!",
            "I hate waiting in long lines."
        ]
        
        # Test VADER
        print("  Testing VADER...")
        for text in test_texts:
            result = analyzer.analyze_with_vader(text)
            print(f"    '{text}' -> {result['sentiment']} ({result['compound_score']:.3f})")
        
        # Test TextBlob
        print("  Testing TextBlob...")
        for text in test_texts:
            result = analyzer.analyze_with_textblob(text)
            print(f"    '{text}' -> {result['sentiment']} ({result['polarity']:.3f})")
        
        # Test Transformer (if available)
        print("  Testing Transformer...")
        for text in test_texts:
            result = analyzer.analyze_with_transformer(text)
            print(f"    '{text}' -> {result['sentiment']} ({result.get('confidence', 0):.3f})")
        
        print("✅ Sentiment analysis test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_analyzer():
    """Test data analysis functionality"""
    print("\n📊 Testing data analysis...")
    try:
        analyzer = MoodAnalyzer()
        
        # Create test data
        test_data = {
            'text': ['I love this', 'This is bad', 'Okay product', 'Amazing!', 'Terrible'],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'keyword': ['love', 'bad', 'okay', 'amazing', 'terrible']
        }
        df = pd.DataFrame(test_data)
        
        # Test sentiment distribution
        distribution = analyzer.calculate_sentiment_distribution(df)
        print("  Sentiment distribution:")
        print(distribution)
        
        # Test metrics
        metrics = analyzer.calculate_sentiment_metrics(df)
        print("  Sentiment metrics:")
        print(f"    Total tweets: {metrics['total_tweets']}")
        print(f"    Dominant sentiment: {metrics['dominant_sentiment']}")
        
        print("✅ Data analysis test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data analysis test failed: {e}")
        return False

def test_twitter_api():
    """Test Twitter API connectivity (without collecting tweets)"""
    print("\n🐦 Testing Twitter API connectivity...")
    try:
        collector = TwitterCollector()
        print("✅ Twitter API client initialized successfully")
        
        # Note: We don't actually make API calls in the test to avoid rate limits
        print("  (Skipping actual API call to avoid rate limits)")
        
        return True
        
    except Exception as e:
        print(f"❌ Twitter API test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test with sample data"""
    print("\n🔬 Running comprehensive test...")
    try:
        # Create sample tweet data
        sample_tweets = [
            {
                'id': '1',
                'text': 'I love this amazing product! #happy #excited',
                'created_at': datetime.now(),
                'author_id': 'user1',
                'keyword': 'love'
            },
            {
                'id': '2', 
                'text': 'This is absolutely terrible and disappointing',
                'created_at': datetime.now(),
                'author_id': 'user2',
                'keyword': 'terrible'
            },
            {
                'id': '3',
                'text': 'The weather is okay today, nothing special',
                'created_at': datetime.now(),
                'author_id': 'user3',
                'keyword': 'weather'
            }
        ]
        
        df = pd.DataFrame(sample_tweets)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_tweets(df)
        
        # Analyze sentiment
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results = sentiment_analyzer.analyze_sentiment_batch(
            df_processed['lemmatized_text'].tolist(),
            method='vader'
        )
        
        # Merge results
        for col in sentiment_results.columns:
            if col != 'text':
                df_processed[col] = sentiment_results[col]
        
        # Analyze results
        mood_analyzer = MoodAnalyzer()
        distribution = mood_analyzer.calculate_sentiment_distribution(df_processed, 'vader_sentiment')
        
        print("  Sample tweet analysis results:")
        print(df_processed[['text', 'lemmatized_text', 'vader_sentiment']].to_string())
        print("\n  Sentiment distribution:")
        print(distribution.to_string())
        
        print("✅ Comprehensive test passed")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Twitter Mood Detection System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Preprocessing", test_data_preprocessing),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Data Analysis", test_analyzer),
        ("Twitter API", test_twitter_api),
        ("Comprehensive Test", run_comprehensive_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


