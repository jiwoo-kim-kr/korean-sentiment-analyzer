# Korean Sentiment Analyzer

A Python tool for analyzing sentiment in Korean text using various NLP techniques and machine learning models.

## Features

- **Multiple Model Types**: Support for basic keyword matching, transformer-based models, and ensemble approaches
- **Korean Language Support**: Optimized for Korean text processing with KoNLPy integration
- **Flexible Input**: Analyze text from command line arguments or files
- **Batch Processing**: Support for analyzing multiple texts at once
- **Confidence Scoring**: Provides confidence scores for sentiment predictions

## Installation

1. Clone the repository:
```bash
cd /home/user/Desktop/korean-sentiment-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Analyze text directly:
```bash
python main.py --text "This is a sample Korean text"
```

Analyze text from a file:
```bash
python main.py --file sample_text.txt
```

Choose different models:
```bash
python main.py --text "Your text here" --model basic
python main.py --text "Your text here" --model transformer
python main.py --text "Your text here" --model ensemble
```

### Python API

```python
from src.sentiment_analyzer import KoreanSentimentAnalyzer

# Create analyzer with basic model
analyzer = KoreanSentimentAnalyzer(model_type='basic')

# Analyze sentiment
result = analyzer.analyze_sentiment("This is a sample text")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")

# Batch analysis
texts = ["Text 1", "Text 2", "Text 3"]
results = analyzer.batch_analyze(texts)
```

## Model Types

### Basic Model
- Uses keyword matching with positive and negative Korean sentiment words
- Fast and lightweight
- Good for simple sentiment analysis tasks

### Transformer Model
- Uses pre-trained Korean BERT models (KLUE/BERT)
- More accurate for complex sentences
- Requires more computational resources

### Ensemble Model
- Combines multiple models for better accuracy
- Uses voting mechanism for final prediction
- Balanced approach between speed and accuracy

## Project Structure

```
korean-sentiment-analyzer/
|-- main.py                 # Main entry point
|-- src/
|   |-- __init__.py        # Package initialization
|   |-- sentiment_analyzer.py  # Core sentiment analysis logic
|-- requirements.txt       # Python dependencies
|-- README.md             # Project documentation
```

## Dependencies

Key dependencies include:
- `numpy`, `pandas`: Data processing
- `konlpy`: Korean text processing
- `torch`, `transformers`: Deep learning models
- `klue`: Korean language understanding evaluation
- `scikit-learn`: Machine learning utilities

## Examples

### Basic Usage
```bash
# Positive sentiment
python main.py --text "I love this product! It's amazing."

# Negative sentiment  
python main.py --text "This is terrible and disappointing."

# File analysis
echo "This is wonderful and great!" > sample.txt
python main.py --file sample.txt
```

### Python Script Example
```python
from src.sentiment_analyzer import KoreanSentimentAnalyzer

# Test different models
models = ['basic', 'transformer', 'ensemble']
text = "This movie was absolutely fantastic and entertaining!"

for model in models:
    analyzer = KoreanSentimentAnalyzer(model_type=model)
    result = analyzer.analyze_sentiment(text)
    print(f"Model: {model}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("---")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Future Enhancements

- Integration with more Korean language models
- Web interface for easy usage
- Real-time sentiment analysis API
- Support for more languages
- Advanced visualization of results
