"""
Sentiment analysis module using multiple approaches
"""
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import os
from typing import List, Dict, Tuple

from config import Config

class SentimentAnalyzer:
    """Class for sentiment analysis using multiple approaches"""
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load pre-trained transformer model
        try:
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=Config.SENTIMENT_MODEL_NAME,
                tokenizer=Config.SENTIMENT_MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            self.transformer_pipeline = None
            
        # Custom model components
        self.tfidf_vectorizer = None
        self.custom_model = None
        
    def analyze_with_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'neutral_score': scores['neu'],
            'negative_score': scores['neg']
        }
    
    def analyze_with_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def analyze_with_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using pre-trained transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.transformer_pipeline is None:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'Transformer model not available'
            }
            
        try:
            # Handle long texts by truncating
            if len(text) > 512:
                text = text[:512]
                
            result = self.transformer_pipeline(text)[0]
            
            # Map transformer labels to our format
            sentiment_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'LABEL_0': 'negative',  # Some models use this format
                'LABEL_1': 'positive'
            }
            
            sentiment = sentiment_mapping.get(result['label'], 'neutral')
            
            return {
                'sentiment': sentiment,
                'confidence': result['score'],
                'label': result['label']
            }
            
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_sentiment_batch(self, texts: List[str], method: str = 'all') -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            method: Analysis method ('vader', 'textblob', 'transformer', 'all')
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing text {i+1}/{len(texts)}")
                
            result = {'text': text}
            
            if method in ['vader', 'all']:
                vader_result = self.analyze_with_vader(text)
                result.update({f'vader_{k}': v for k, v in vader_result.items()})
                
            if method in ['textblob', 'all']:
                textblob_result = self.analyze_with_textblob(text)
                result.update({f'textblob_{k}': v for k, v in textblob_result.items()})
                
            if method in ['transformer', 'all']:
                transformer_result = self.analyze_with_transformer(text)
                result.update({f'transformer_{k}': v for k, v in transformer_result.items()})
                
            results.append(result)
            
        return pd.DataFrame(results)
    
    def train_custom_model(self, df: pd.DataFrame, text_column: str = 'lemmatized_text', 
                          label_column: str = 'sentiment') -> Dict:
        """
        Train a custom sentiment classification model
        
        Args:
            df: DataFrame with text and labels
            text_column: Column containing text data
            label_column: Column containing sentiment labels
            
        Returns:
            Dictionary with model performance metrics
        """
        print("Training custom sentiment model...")
        
        # Prepare data
        X = df[text_column].fillna('')
        y = df[label_column]
        
        # Vectorize text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_vectorized = self.tfidf_vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.custom_model = LogisticRegression(random_state=42, max_iter=1000)
        self.custom_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.custom_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Custom model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.save_custom_model()
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_with_custom_model(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment using custom trained model
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of predicted sentiment labels
        """
        if self.custom_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Custom model not trained. Call train_custom_model first.")
            
        # Vectorize texts
        X_vectorized = self.tfidf_vectorizer.transform(texts)
        
        # Predict
        predictions = self.custom_model.predict(X_vectorized)
        
        return predictions.tolist()
    
    def save_custom_model(self):
        """Save the custom trained model"""
        import joblib
        
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        model_path = os.path.join(Config.MODELS_DIR, 'custom_sentiment_model.joblib')
        vectorizer_path = os.path.join(Config.MODELS_DIR, 'tfidf_vectorizer.joblib')
        
        joblib.dump(self.custom_model, model_path)
        joblib.dump(self.tfidf_vectorizer, vectorizer_path)
        
        print(f"Custom model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_custom_model(self):
        """Load the custom trained model"""
        import joblib
        
        model_path = os.path.join(Config.MODELS_DIR, 'custom_sentiment_model.joblib')
        vectorizer_path = os.path.join(Config.MODELS_DIR, 'tfidf_vectorizer.joblib')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.custom_model = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(vectorizer_path)
            print("Custom model loaded successfully")
        else:
            print("Custom model files not found")


