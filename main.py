#!/usr/bin/env python3
"""
Korean Sentiment Analyzer
A tool for analyzing sentiment in Korean text using various NLP techniques.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.sentiment_analyzer import KoreanSentimentAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment in Korean text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--model', type=str, default='basic', 
                       choices=['basic', 'transformer', 'ensemble'],
                       help='Model type to use for analysis')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        print("Error: Please provide either --text or --file argument")
        sys.exit(1)
    
    analyzer = KoreanSentimentAnalyzer(model_type=args.model)
    
    if args.text:
        result = analyzer.analyze_sentiment(args.text)
        print(f"Text: {args.text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            result = analyzer.analyze_sentiment(text)
            print(f"File: {args.file}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
        except FileNotFoundError:
            print(f"Error: File {args.file} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
