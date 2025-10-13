#!/usr/bin/env python3
"""
Quick test script to verify the Twitter Mood Detection System works
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality without NLTK dependencies"""
    print("🧪 Quick Test - Twitter Mood Detection System")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from config import Config
        from twitter_collector import TwitterCollector
        from sentiment_analyzer import SentimentAnalyzer
        print("✅ All core modules imported successfully")
        
        # Test configuration
        print("\n🔧 Testing configuration...")
        try:
            Config.validate_config()
            print("✅ Configuration valid")
        except ValueError as e:
            print(f"⚠️  Configuration warning: {e}")
            print("   (This is expected if you haven't set up your Twitter API token yet)")
        
        # Test sentiment analysis
        print("\n🧠 Testing sentiment analysis...")
        analyzer = SentimentAnalyzer()
        
        test_texts = [
            "I love this amazing product!",
            "This is absolutely terrible.",
            "The weather is okay today."
        ]
        
        for text in test_texts:
            # Test VADER
            vader_result = analyzer.analyze_with_vader(text)
            print(f"  VADER: '{text}' -> {vader_result['sentiment']} ({vader_result['compound_score']:.3f})")
            
            # Test TextBlob
            textblob_result = analyzer.analyze_with_textblob(text)
            print(f"  TextBlob: '{text}' -> {textblob_result['sentiment']} ({textblob_result['polarity']:.3f})")
        
        print("✅ Sentiment analysis working!")
        
        # Test Twitter API client (without making actual calls)
        print("\n🐦 Testing Twitter API client...")
        try:
            collector = TwitterCollector()
            print("✅ Twitter API client initialized successfully")
        except Exception as e:
            print(f"⚠️  Twitter API warning: {e}")
            print("   (This is expected if you haven't set up your API token yet)")
        
        print("\n🎉 Quick test completed successfully!")
        print("\n📋 Next steps:")
        print("1. Set your TWITTER_BEARER_TOKEN in the .env file")
        print("2. Run: python run_dashboard.py")
        print("3. Or test with: cd src && python example_usage.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()


