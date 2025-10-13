"""
Data preprocessing module for Twitter mood detection
"""
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List
import unicodedata

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class DataPreprocessor:
    """Class for preprocessing tweet data"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Warning: NLTK stopwords not found. Using basic stopwords.")
            # Basic stopwords if NLTK data not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except (LookupError, Exception):
            print("Warning: NLTK wordnet not found. Using basic lemmatization.")
            self.lemmatizer = None
        
        # Additional stop words for Twitter
        self.stop_words.update(['rt', 'amp', 'http', 'https', 'www', 'com'])
        
    def clean_text(self, text: str) -> str:
        """
        Clean tweet text by removing URLs, mentions, hashtags, etc.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove emojis and special unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Remove extra whitespace again
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple splitting if NLTK tokenizer not available
            tokens = text.split()
        
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using NLTK's WordNetLemmatizer
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple splitting if NLTK tokenizer not available
            tokens = text.split()
        
        if self.lemmatizer is not None:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            # Fallback to basic lemmatization (just lowercase)
            lemmatized_tokens = [token.lower() for token in tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def preprocess_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire DataFrame of tweets
        
        Args:
            df: DataFrame with tweet data
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        print("Cleaning tweet text...")
        df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        print("Removing stopwords...")
        df_processed['no_stopwords'] = df_processed['cleaned_text'].apply(self.remove_stopwords)
        
        print("Lemmatizing text...")
        df_processed['lemmatized_text'] = df_processed['no_stopwords'].apply(self.lemmatize_text)
        
        # Remove empty tweets after preprocessing
        df_processed = df_processed[df_processed['lemmatized_text'].str.len() > 0]
        
        print(f"Preprocessing complete. {len(df_processed)} tweets remaining after cleaning.")
        
        return df_processed
    
    def get_word_frequency(self, df: pd.DataFrame, text_column: str = 'lemmatized_text') -> pd.Series:
        """
        Get word frequency from preprocessed text
        
        Args:
            df: DataFrame with preprocessed tweets
            text_column: Column containing processed text
            
        Returns:
            Series with word frequencies
        """
        all_text = ' '.join(df[text_column].astype(str))
        
        try:
            tokens = word_tokenize(all_text)
        except LookupError:
            # Fallback to simple splitting if NLTK tokenizer not available
            tokens = all_text.split()
        
        # Filter out single characters and empty strings
        tokens = [token for token in tokens if len(token) > 1]
        
        # Count frequencies
        word_freq = pd.Series(tokens).value_counts()
        
        return word_freq
