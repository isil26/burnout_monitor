"""Sentiment analysis with multiple model support."""

import streamlit as st
from textblob import TextBlob
from config import (
    SENTIMENT_MODEL,
    SENTIMENT_STRESSED_THRESHOLD,
    SENTIMENT_HEALTHY_THRESHOLD,
    SENTIMENT_SEVERE_STRESS_THRESHOLD
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    _vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    _vader_analyzer = None


@st.cache_data(show_spinner=False)
def calculate_sentiment_textblob(text):
    """Calculate sentiment using TextBlob."""
    try:
        if not text or len(str(text).strip()) == 0:
            return 0.0
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except:
        return 0.0


@st.cache_data(show_spinner=False)
def calculate_sentiment_vader(text):
    """Calculate sentiment using VADER (optimized for social media and short text)."""
    try:
        if not text or len(str(text).strip()) == 0:
            return 0.0
        if not VADER_AVAILABLE or _vader_analyzer is None:
            return calculate_sentiment_textblob(text)
        score = _vader_analyzer.polarity_scores(str(text))['compound']
        return score
    except:
        return 0.0


@st.cache_data(show_spinner=False)
def calculate_sentiment(text, model=None):
    """Calculate sentiment polarity score (-1.0 to 1.0)."""
    if model is None:
        model = SENTIMENT_MODEL
    
    if model == 'vader' and VADER_AVAILABLE:
        return calculate_sentiment_vader(text)
    else:
        return calculate_sentiment_textblob(text)


def classify_sentiment(score):
    """Classify sentiment as Stressed, Neutral, or Healthy."""
    if score < SENTIMENT_STRESSED_THRESHOLD:
        return "Stressed"
    elif score > SENTIMENT_HEALTHY_THRESHOLD:
        return "Healthy"
    else:
        return "Neutral"


def get_sentiment_emoji(score):
    """Return color indicator for sentiment score."""
    if score < SENTIMENT_STRESSED_THRESHOLD:
        return ""
    elif score > SENTIMENT_HEALTHY_THRESHOLD:
        return ""
    else:
        return ""


def get_risk_level(score):
    """Return risk level label based on sentiment."""
    if score < SENTIMENT_SEVERE_STRESS_THRESHOLD:
        return " CRITICAL"
    elif score < SENTIMENT_STRESSED_THRESHOLD:
        return " HIGH"
    elif score < 0:
        return " MEDIUM"
    else:
        return " LOW"


def add_sentiment_analysis(df, model=None):
    """Add sentiment column to dataframe."""
    if model is None:
        model = SENTIMENT_MODEL
    
    model_name = "VADER" if model == 'vader' and VADER_AVAILABLE else "TextBlob"
    st.info(f"Analyzing sentiment using {model_name}... This may take a moment.")
    df['sentiment'] = df['body'].apply(lambda x: calculate_sentiment(x, model))
    return df


def get_available_models():
    """Return list of available sentiment models."""
    models = ['textblob']
    if VADER_AVAILABLE:
        models.append('vader')
    return models


def calculate_sentiment_statistics(df):
    """Calculate sentiment distribution and health score."""
    negative = len(df[df['sentiment'] < SENTIMENT_STRESSED_THRESHOLD])
    neutral = len(df[(df['sentiment'] >= SENTIMENT_STRESSED_THRESHOLD) & 
                     (df['sentiment'] <= SENTIMENT_HEALTHY_THRESHOLD)])
    positive = len(df[df['sentiment'] > SENTIMENT_HEALTHY_THRESHOLD])
    
    total = len(df)
    
    return {
        'negative_count': negative,
        'neutral_count': neutral,
        'positive_count': positive,
        'negative_pct': negative / total * 100 if total > 0 else 0,
        'neutral_pct': neutral / total * 100 if total > 0 else 0,
        'positive_pct': positive / total * 100 if total > 0 else 0,
        'mean': df['sentiment'].mean(),
        'median': df['sentiment'].median(),
        'std': df['sentiment'].std(),
        'q1': df['sentiment'].quantile(0.25),
        'q3': df['sentiment'].quantile(0.75),
        'health_score': ((positive - negative) / total) * 100 if total > 0 else 0
    }
