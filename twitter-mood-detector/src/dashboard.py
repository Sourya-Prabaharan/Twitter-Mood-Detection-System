"""
Streamlit dashboard for Twitter Mood Detection System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from twitter_collector import TwitterCollector
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from analyzer import MoodAnalyzer

# Page configuration
st.set_page_config(
    page_title=Config.DASHBOARD_TITLE,
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        color: #2E8B57;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #DC143C;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #4682B4;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🐦 Twitter Mood Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Check if API token is configured
    if not Config.TWITTER_BEARER_TOKEN or Config.TWITTER_BEARER_TOKEN == 'your_bearer_token_here':
        st.error("⚠️ Twitter API Bearer Token not configured!")
        st.sidebar.info("Please set your TWITTER_BEARER_TOKEN in the .env file")
        st.stop()
    
    # Initialize components
    try:
        collector = TwitterCollector()
        preprocessor = DataPreprocessor()
        sentiment_analyzer = SentimentAnalyzer()
        mood_analyzer = MoodAnalyzer()
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.stop()
    
    # Sidebar options
    st.sidebar.header("📊 Analysis Options")
    
    # Method selection
    analysis_method = st.sidebar.selectbox(
        "Sentiment Analysis Method",
        ["VADER", "TextBlob", "Transformer", "All Methods"]
    )
    
    # Keyword input
    st.sidebar.header("🔍 Tweet Collection")
    keywords_input = st.sidebar.text_input(
        "Enter keywords (comma-separated)",
        value="happy, sad, excited, angry, love, hate"
    )
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    
    max_tweets = st.sidebar.slider(
        "Maximum tweets per keyword",
        min_value=50,
        max_value=500,
        value=100
    )
    
    # Load existing data option
    load_existing = st.sidebar.checkbox("Load existing data file")
    
    if load_existing:
        data_files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith('.csv')]
        if data_files:
            selected_file = st.sidebar.selectbox("Select data file", data_files)
        else:
            st.sidebar.warning("No data files found")
            load_existing = False
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Data Collection", "📊 Analysis", "📝 Reports"])
    
    with tab1:
        st.header("📈 System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Available Keywords", len(keywords))
        
        with col2:
            st.metric("Max Tweets per Keyword", max_tweets)
        
        with col3:
            st.metric("Analysis Method", analysis_method)
        
        # Quick stats if data exists
        if load_existing and 'selected_file' in locals():
            try:
                df = pd.read_csv(os.path.join(Config.DATA_DIR, selected_file))
                st.success(f"📁 Loaded {len(df)} tweets from {selected_file}")
                
                # Quick sentiment overview
                if 'sentiment' in df.columns:
                    sentiment_counts = df['sentiment'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Tweets", sentiment_counts.get('positive', 0))
                    with col2:
                        st.metric("Negative Tweets", sentiment_counts.get('negative', 0))
                    with col3:
                        st.metric("Neutral Tweets", sentiment_counts.get('neutral', 0))
                        
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with tab2:
        st.header("🔍 Tweet Collection")
        
        if st.button("🚀 Collect New Tweets", type="primary"):
            if not keywords:
                st.error("Please enter at least one keyword")
            else:
                with st.spinner("Collecting tweets..."):
                    try:
                        # Collect tweets
                        tweets_df = collector.collect_mood_tweets(keywords, max_tweets)
                        
                        if not tweets_df.empty:
                            # Preprocess tweets
                            tweets_df = preprocessor.preprocess_tweets(tweets_df)
                            
                            # Save tweets
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"tweets_{timestamp}.csv"
                            collector.save_tweets_to_csv(tweets_df, filename)
                            
                            st.success(f"✅ Collected and saved {len(tweets_df)} tweets!")
                            st.dataframe(tweets_df.head())
                            
                            # Store in session state
                            st.session_state['tweets_df'] = tweets_df
                        else:
                            st.warning("No tweets collected. Try different keywords or check your API credentials.")
                            
                    except Exception as e:
                        st.error(f"Error collecting tweets: {e}")
        
        # Load existing data
        if load_existing and 'selected_file' in locals():
            if st.button("📂 Load Selected File"):
                try:
                    df = pd.read_csv(os.path.join(Config.DATA_DIR, selected_file))
                    st.session_state['tweets_df'] = df
                    st.success(f"Loaded {len(df)} tweets from {selected_file}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    
    with tab3:
        st.header("📊 Sentiment Analysis")
        
        if 'tweets_df' not in st.session_state:
            st.info("👆 Please collect tweets first or load existing data")
        else:
            df = st.session_state['tweets_df']
            
            # Perform sentiment analysis
            if st.button("🧠 Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Select analysis method
                        method_map = {
                            "VADER": "vader",
                            "TextBlob": "textblob", 
                            "Transformer": "transformer",
                            "All Methods": "all"
                        }
                        
                        method = method_map[analysis_method]
                        
                        # Analyze sentiment
                        sentiment_results = sentiment_analyzer.analyze_sentiment_batch(
                            df['lemmatized_text'].tolist(),
                            method=method
                        )
                        
                        # Merge results with original data
                        for col in sentiment_results.columns:
                            if col != 'text':
                                df[col] = sentiment_results[col]
                        
                        # Update session state
                        st.session_state['tweets_df'] = df
                        
                        st.success("✅ Sentiment analysis complete!")
                        
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {e}")
            
            # Display analysis results
            if 'tweets_df' in st.session_state:
                df = st.session_state['tweets_df']
                
                # Check if sentiment analysis has been performed
                sentiment_columns = [col for col in df.columns if 'sentiment' in col]
                
                if sentiment_columns:
                    # Select primary sentiment column
                    primary_sentiment_col = st.selectbox(
                        "Select sentiment column for analysis",
                        sentiment_columns
                    )
                    
                    # Overview metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_tweets = len(df)
                    sentiment_counts = df[primary_sentiment_col].value_counts()
                    
                    with col1:
                        st.metric("Total Tweets", total_tweets)
                    with col2:
                        st.metric("Positive", sentiment_counts.get('positive', 0))
                    with col3:
                        st.metric("Negative", sentiment_counts.get('negative', 0))
                    with col4:
                        st.metric("Neutral", sentiment_counts.get('neutral', 0))
                    
                    # Visualizations
                    st.subheader("📊 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution pie chart
                        fig_pie = mood_analyzer.plot_sentiment_distribution(df, primary_sentiment_col)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Sentiment by keyword
                        if 'keyword' in df.columns:
                            fig_bar = mood_analyzer.plot_keyword_sentiment(df, primary_sentiment_col)
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Timeline if time data exists
                    if 'created_at' in df.columns:
                        st.subheader("📅 Timeline Analysis")
                        fig_timeline = mood_analyzer.plot_sentiment_timeline(df, primary_sentiment_col)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Word clouds
                    st.subheader("☁️ Word Clouds by Sentiment")
                    if st.button("Generate Word Clouds"):
                        fig_wc, wordclouds = mood_analyzer.plot_word_clouds_by_sentiment(df)
                        st.pyplot(fig_wc)
                    
                    # Method comparison
                    if len(sentiment_columns) > 1:
                        st.subheader("🔄 Method Comparison")
                        fig_comparison = mood_analyzer.compare_sentiment_methods(df)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                
                else:
                    st.info("No sentiment analysis results found. Please run sentiment analysis first.")
    
    with tab4:
        st.header("📝 Analysis Reports")
        
        if 'tweets_df' not in st.session_state:
            st.info("👆 Please collect tweets and run analysis first")
        else:
            df = st.session_state['tweets_df']
            sentiment_columns = [col for col in df.columns if 'sentiment' in col]
            
            if sentiment_columns:
                # Select sentiment column for report
                report_sentiment_col = st.selectbox(
                    "Select sentiment column for report",
                    sentiment_columns,
                    key="report_sentiment"
                )
                
                # Generate comprehensive report
                if st.button("📋 Generate Report"):
                    with st.spinner("Generating report..."):
                        try:
                            report = mood_analyzer.generate_summary_report(df, report_sentiment_col)
                            
                            st.subheader("📊 Summary Statistics")
                            
                            # Basic metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.json(report['basic_metrics'])
                            
                            with col2:
                                st.json(report['distribution'])
                            
                            # Export options
                            st.subheader("💾 Export Options")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Export CSV
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📄 Download CSV",
                                    data=csv,
                                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Export JSON
                                json_data = report
                                st.download_button(
                                    label="📋 Download Report (JSON)",
                                    data=pd.Series(json_data).to_json(),
                                    file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            
                            with col3:
                                # Display sample data
                                if st.button("👀 View Sample Data"):
                                    st.dataframe(df.sample(min(100, len(df))))
                                    
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
            else:
                st.info("No sentiment analysis results found. Please run sentiment analysis first.")

if __name__ == "__main__":
    main()

