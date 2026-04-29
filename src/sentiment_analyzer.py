"""
Korean Sentiment Analyzer Module
Contains the main sentiment analysis logic for Korean text.
"""

import re
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod


class SentimentModel(ABC):
    """Abstract base class for sentiment analysis models."""
    
    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment and confidence score."""
        pass


class BasicSentimentModel(SentimentModel):
    """Basic sentiment analysis using keyword matching."""
    
    def __init__(self):
        # Positive and negative Korean sentiment keywords
        self.positive_keywords = {
            'good': ['good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good'],
            'happy': ['happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy'],
            'love': ['love', 'love', 'love', 'love', 'love', 'love', 'love', 'love', 'love', 'love'],
            'great': ['great', 'great', 'great', 'great', 'great', 'great', 'great', 'great', 'great', 'great'],
            'best': ['best', 'best', 'best', 'best', 'best', 'best', 'best', 'best', 'best', 'best'],
            'excellent': ['excellent', 'excellent', 'excellent', 'excellent', 'excellent', 'excellent', 'excellent', 'excellent', 'excellent', 'excellent'],
            'wonderful': ['wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful', 'wonderful'],
            'amazing': ['amazing', 'amazing', 'amazing', 'amazing', 'amazing', 'amazing', 'amazing', 'amazing', 'amazing', 'amazing'],
            'perfect': ['perfect', 'perfect', 'perfect', 'perfect', 'perfect', 'perfect', 'perfect', 'perfect', 'perfect', 'perfect'],
            'awesome': ['awesome', 'awesome', 'awesome', 'awesome', 'awesome', 'awesome', 'awesome', 'awesome', 'awesome', 'awesome']
        }
        
        self.negative_keywords = {
            'bad': ['bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad'],
            'sad': ['sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad'],
            'hate': ['hate', 'hate', 'hate', 'hate', 'hate', 'hate', 'hate', 'hate', 'hate', 'hate'],
            'terrible': ['terrible', 'terrible', 'terrible', 'terrible', 'terrible', 'terrible', 'terrible', 'terrible', 'terrible', 'terrible'],
            'worst': ['worst', 'worst', 'worst', 'worst', 'worst', 'worst', 'worst', 'worst', 'worst', 'worst'],
            'awful': ['awful', 'awful', 'awful', 'awful', 'awful', 'awful', 'awful', 'awful', 'awful', 'awful'],
            'horrible': ['horrible', 'horrible', 'horrible', 'horrible', 'horrible', 'horrible', 'horrible', 'horrible', 'horrible', 'horrible'],
            'disgusting': ['disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting', 'disgusting'],
            'disappointing': ['disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing', 'disappointing'],
            'annoying': ['annoying', 'annoying', 'annoying', 'annoying', 'annoying', 'annoying', 'annoying', 'annoying', 'annoying', 'annoying']
        }
        
        # Flatten keyword lists for easier matching
        self.positive_words = []
        for category, words in self.positive_keywords.items():
            self.positive_words.extend(words)
        
        self.negative_words = []
        for category, words in self.negative_keywords.items():
            self.negative_words.extend(words)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using keyword matching."""
        text_lower = text.lower()
        
        positive_count = 0
        negative_count = 0
        
        # Count positive and negative words
        for word in self.positive_words:
            positive_count += text_lower.count(word)
        
        for word in self.negative_words:
            negative_count += text_lower.count(word)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return sentiment, confidence


class TransformerSentimentModel(SentimentModel):
    """Transformer-based sentiment analysis model."""
    
    def __init__(self):
        # Placeholder for transformer model initialization
        # In a real implementation, this would load a pre-trained Korean model
        self.model_name = "klue/bert-base"
        print(f"Note: Transformer model {self.model_name} would be loaded here")
        print("Using fallback to basic model for demonstration")
        
        # Use basic model as fallback
        self.basic_model = BasicSentimentModel()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using transformer model."""
        # In a real implementation, this would use the actual transformer model
        # For now, fall back to basic model
        return self.basic_model.predict(text)


class EnsembleSentimentModel(SentimentModel):
    """Ensemble model combining multiple approaches."""
    
    def __init__(self):
        self.models = [
            BasicSentimentModel(),
            TransformerSentimentModel()
        ]
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using ensemble of models."""
        predictions = []
        confidences = []
        
        for model in self.models:
            sentiment, confidence = model.predict(text)
            predictions.append(sentiment)
            confidences.append(confidence)
        
        # Simple voting mechanism
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment in predictions:
            sentiment_counts[sentiment] += 1
        
        # Determine final sentiment
        max_count = max(sentiment_counts.values())
        top_sentiments = [s for s, c in sentiment_counts.items() if c == max_count]
        
        if len(top_sentiments) == 1:
            final_sentiment = top_sentiments[0]
        else:
            # In case of tie, use the one with higher average confidence
            sentiment_confidences = {}
            for i, sentiment in enumerate(predictions):
                if sentiment not in sentiment_confidences:
                    sentiment_confidences[sentiment] = []
                sentiment_confidences[sentiment].append(confidences[i])
            
            avg_confidences = {s: sum(c) / len(c) for s, c in sentiment_confidences.items()}
            final_sentiment = max(avg_confidences, key=avg_confidences.get)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return final_sentiment, avg_confidence


class KoreanSentimentAnalyzer:
    """Main sentiment analyzer class."""
    
    def __init__(self, model_type: str = 'basic'):
        self.model_type = model_type
        self.model = self._create_model(model_type)
    
    def _create_model(self, model_type: str) -> SentimentModel:
        """Create the appropriate model based on type."""
        if model_type == 'basic':
            return BasicSentimentModel()
        elif model_type == 'transformer':
            return TransformerSentimentModel()
        elif model_type == 'ensemble':
            return EnsembleSentimentModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment of the given text."""
        if not text or not text.strip():
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        sentiment, confidence = self.model.predict(text)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'model_type': self.model_type
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, any]]:
        """Analyze sentiment for multiple texts."""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
