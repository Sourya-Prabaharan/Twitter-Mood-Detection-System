"""
Data analysis and visualization module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple

class MoodAnalyzer:
    """Class for analyzing and visualizing mood data"""
    
    def __init__(self):
        """Initialize the analyzer"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def calculate_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """
        Calculate sentiment distribution
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            
        Returns:
            DataFrame with sentiment distribution
        """
        distribution = df[sentiment_column].value_counts()
        percentage = df[sentiment_column].value_counts(normalize=True) * 100
        
        result = pd.DataFrame({
            'count': distribution,
            'percentage': percentage
        }).reset_index()
        
        result.columns = ['sentiment', 'count', 'percentage']
        
        return result
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> go.Figure:
        """
        Create interactive pie chart of sentiment distribution
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            
        Returns:
            Plotly figure
        """
        distribution = self.calculate_sentiment_distribution(df, sentiment_column)
        
        fig = px.pie(
            distribution,
            values='count',
            names='sentiment',
            title='Sentiment Distribution',
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#4682B4'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    
    def plot_sentiment_timeline(self, df: pd.DataFrame, 
                               sentiment_column: str = 'sentiment',
                               time_column: str = 'created_at') -> go.Figure:
        """
        Create timeline plot of sentiment over time
        
        Args:
            df: DataFrame with sentiment and time data
            sentiment_column: Column containing sentiment labels
            time_column: Column containing timestamps
            
        Returns:
            Plotly figure
        """
        # Convert time column to datetime
        df_time = df.copy()
        df_time[time_column] = pd.to_datetime(df_time[time_column])
        
        # Group by date and sentiment
        df_time['date'] = df_time[time_column].dt.date
        timeline_data = df_time.groupby(['date', sentiment_column]).size().reset_index(name='count')
        
        # Create subplot
        fig = make_subplots(rows=1, cols=1)
        
        sentiments = timeline_data[sentiment_column].unique()
        colors = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}
        
        for sentiment in sentiments:
            sentiment_data = timeline_data[timeline_data[sentiment_column] == sentiment]
            
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['date'],
                    y=sentiment_data['count'],
                    mode='lines+markers',
                    name=sentiment.title(),
                    line=dict(color=colors.get(sentiment, '#000000'))
                )
            )
        
        fig.update_layout(
            title='Sentiment Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Tweets',
            height=500
        )
        
        return fig
    
    def plot_keyword_sentiment(self, df: pd.DataFrame,
                              sentiment_column: str = 'sentiment',
                              keyword_column: str = 'keyword') -> go.Figure:
        """
        Create bar chart showing sentiment by keyword
        
        Args:
            df: DataFrame with sentiment and keyword data
            sentiment_column: Column containing sentiment labels
            keyword_column: Column containing keywords
            
        Returns:
            Plotly figure
        """
        # Create crosstab
        crosstab = pd.crosstab(df[keyword_column], df[sentiment_column])
        
        fig = px.bar(
            crosstab,
            title='Sentiment Distribution by Keyword',
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title='Keyword',
            yaxis_title='Number of Tweets',
            height=500
        )
        
        return fig
    
    def create_word_cloud(self, df: pd.DataFrame, 
                         text_column: str = 'lemmatized_text',
                         sentiment_column: str = 'sentiment',
                         sentiment_filter: str = None) -> WordCloud:
        """
        Create word cloud for tweets
        
        Args:
            df: DataFrame with text data
            text_column: Column containing processed text
            sentiment_column: Column containing sentiment labels
            sentiment_filter: Filter by specific sentiment (optional)
            
        Returns:
            WordCloud object
        """
        # Filter by sentiment if specified
        if sentiment_filter:
            filtered_df = df[df[sentiment_column] == sentiment_filter]
        else:
            filtered_df = df
            
        # Combine all text
        all_text = ' '.join(filtered_df[text_column].fillna(''))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        return wordcloud
    
    def plot_word_clouds_by_sentiment(self, df: pd.DataFrame,
                                     text_column: str = 'lemmatized_text',
                                     sentiment_column: str = 'sentiment') -> Tuple[plt.Figure, List[WordCloud]]:
        """
        Create word clouds for each sentiment
        
        Args:
            df: DataFrame with text and sentiment data
            text_column: Column containing processed text
            sentiment_column: Column containing sentiment labels
            
        Returns:
            Tuple of matplotlib figure and list of word clouds
        """
        sentiments = df[sentiment_column].unique()
        n_sentiments = len(sentiments)
        
        fig, axes = plt.subplots(1, n_sentiments, figsize=(5*n_sentiments, 5))
        if n_sentiments == 1:
            axes = [axes]
            
        wordclouds = []
        
        for i, sentiment in enumerate(sentiments):
            wordcloud = self.create_word_cloud(df, text_column, sentiment_column, sentiment)
            wordclouds.append(wordcloud)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment.title()} Sentiment')
            axes[i].axis('off')
            
        plt.tight_layout()
        
        return fig, wordclouds
    
    def calculate_sentiment_metrics(self, df: pd.DataFrame, 
                                   sentiment_column: str = 'sentiment') -> Dict:
        """
        Calculate various sentiment metrics
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            
        Returns:
            Dictionary with metrics
        """
        total_tweets = len(df)
        sentiment_counts = df[sentiment_column].value_counts()
        
        metrics = {
            'total_tweets': total_tweets,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': (sentiment_counts / total_tweets * 100).to_dict(),
            'dominant_sentiment': sentiment_counts.index[0],
            'sentiment_ratio': sentiment_counts.max() / sentiment_counts.min() if len(sentiment_counts) > 1 else 1.0
        }
        
        return metrics
    
    def compare_sentiment_methods(self, df: pd.DataFrame) -> go.Figure:
        """
        Compare results from different sentiment analysis methods
        
        Args:
            df: DataFrame with sentiment analysis results from multiple methods
            
        Returns:
            Plotly figure
        """
        methods = ['vader_sentiment', 'textblob_sentiment', 'transformer_sentiment']
        available_methods = [col for col in methods if col in df.columns]
        
        if not available_methods:
            raise ValueError("No sentiment analysis columns found in DataFrame")
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=len(available_methods),
            subplot_titles=[method.replace('_sentiment', '').title() for method in available_methods]
        )
        
        for i, method in enumerate(available_methods):
            distribution = df[method].value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=distribution.index,
                    y=distribution.values,
                    name=method.replace('_sentiment', '').title()
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Sentiment Analysis Comparison',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def generate_summary_report(self, df: pd.DataFrame,
                               sentiment_column: str = 'sentiment') -> Dict:
        """
        Generate a comprehensive summary report
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.calculate_sentiment_distribution(df, sentiment_column)
        sentiment_metrics = self.calculate_sentiment_metrics(df, sentiment_column)
        
        # Additional metrics
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            time_span = (df['created_at'].max() - df['created_at'].min()).days
        else:
            time_span = None
            
        report = {
            'basic_metrics': sentiment_metrics,
            'distribution': metrics.to_dict('records'),
            'time_span_days': time_span,
            'average_tweets_per_day': sentiment_metrics['total_tweets'] / time_span if time_span else None
        }
        
        return report


