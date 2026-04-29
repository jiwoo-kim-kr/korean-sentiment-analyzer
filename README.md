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

Analyze Korean text directly:
```bash
python main.py --text "이 제품은 정말 좋아요!"
```

Analyze English text:
```bash
python main.py --text "This product is really great!"
```

Analyze text from a file:
```bash
python main.py --file sample_text.txt
```

Choose different models:
```bash
python main.py --text "이 영화는 정말 재미있어요" --model basic
python main.py --text "이 영화는 정말 재미있어요" --model transformer
python main.py --text "이 영화는 정말 재미있어요" --model ensemble
```

### Python API

```python
from src.sentiment_analyzer import KoreanSentimentAnalyzer

# Create analyzer with basic model
analyzer = KoreanSentimentAnalyzer(model_type='basic')

# Analyze Korean sentiment
result = analyzer.analyze_sentiment("이 제품은 정말 훌륭하고 만족스러워요")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")

# Analyze English sentiment
result = analyzer.analyze_sentiment("This product is excellent and satisfying")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")

# Batch analysis with Korean texts
texts = ["이 영화는 정말 재미있어요", "음식이 맛이 없네요", "그냥 그랬어요"]
results = analyzer.batch_analyze(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {texts[i]}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("---")
```

## Model Types

### Basic Model
- Uses keyword matching with positive and negative Korean sentiment words
- Fast and lightweight
- Good for simple sentiment analysis tasks
- **Best for**: Quick Korean text analysis, simple sentences, prototyping

### Transformer Model
- Uses pre-trained Korean BERT models (KLUE/BERT)
- More accurate for complex sentences and nuanced Korean expressions
- Requires more computational resources
- **Best for**: Complex Korean sentences, formal text, high-accuracy requirements

### Ensemble Model
- Combines multiple models for better accuracy
- Uses voting mechanism for final prediction
- Balanced approach between speed and accuracy
- **Best for**: Production use, mixed language text, reliability requirements

## Korean Language Support

### Supported Korean Text Types
- **Formal Korean**: 제품의 품질에 만족하십니다.
- **Informal Korean**: 이거 진짜 좋아요!
- **Mixed Language**: 이 product는 정말 awesome해요!
- **Slang/Colloquial**: 이거 꿀잼이네요 ㅋㅋㅋ

### Korean Sentiment Keywords
The basic model recognizes common Korean sentiment words:

**Positive Keywords:**
- 좋아요 (good), 최고 (best), 훌륭해요 (excellent)
- 만족스러워요 (satisfying), 재미있어요 (fun/interesting)
- 사랑해요 (love), 행복해요 (happy), 기뻐요 (glad)

**Negative Keywords:**
- 나빠요 (bad), 최악 (worst), 실망했어요 (disappointed)
- 싫어요 (hate), 슬퍼요 (sad), 화나요 (angry)
- 불만이에요 (complaint), 별로예요 (not good)

### Text Encoding
- All Korean text should be in UTF-8 encoding
- Command line arguments support Korean characters
- File input requires UTF-8 encoded text files

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

### Korean Text Examples

#### Command Line Examples
```bash
# Positive Korean sentiment
python main.py --text "이 제품은 정말 좋아요! 최고예요!"

# Negative Korean sentiment
python main.py --text "이 서비스는 최악이에요. 실망했어요."

# Neutral Korean sentiment
python main.py --text "오늘 날씨는 그냥 그래요."

# File analysis with Korean text
echo "한국어 텍스트 분석을 테스트해보세요" > korean_sample.txt
python main.py --file korean_sample.txt
```

#### Mixed Language Examples
```bash
# Mixed Korean and English
python main.py --text "이 product는 정말 awesome해요!"

# Formal Korean
python main.py --text "제품의 품질에 매우 만족하십니다."

# Informal Korean
python main.py --text "이거 진짜 꿀잼이네요 ㅋㅋㅋ"
```

### Basic English Examples
```bash
# Positive sentiment
python main.py --text "I love this product! It's amazing."

# Negative sentiment  
python main.py --text "This is terrible and disappointing."

# File analysis
echo "This is wonderful and great!" > sample.txt
python main.py --file sample.txt
```

### Python Script Examples

#### Korean Text Analysis
```python
from src.sentiment_analyzer import KoreanSentimentAnalyzer

# Test different models with Korean text
models = ['basic', 'transformer', 'ensemble']
korean_text = "이 영화는 정말 재미있고 감동적이었어요!"

for model in models:
    analyzer = KoreanSentimentAnalyzer(model_type=model)
    result = analyzer.analyze_sentiment(korean_text)
    print(f"Model: {model}")
    print(f"Text: {korean_text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("---")
```

#### Comparative Analysis
```python
from src.sentiment_analyzer import KoreanSentimentAnalyzer

# Compare Korean and English sentiment
analyzer = KoreanSentimentAnalyzer(model_type='basic')

korean_texts = [
    "이 제품은 정말 훌륭해요",
    "서비스가 별로 좋지 않았어요",
    "그냥 평범한 제품이에요"
]

english_texts = [
    "This product is excellent",
    "The service was not good",
    "It's just an average product"
]

print("=== Korean Text Analysis ===")
for text in korean_texts:
    result = analyzer.analyze_sentiment(text)
    print(f"'{text}' -> {result['sentiment']} ({result['confidence']:.3f})")

print("\n=== English Text Analysis ===")
for text in english_texts:
    result = analyzer.analyze_sentiment(text)
    print(f"'{text}' -> {result['sentiment']} ({result['confidence']:.3f})")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Future Enhancements

- Integration with more Korean language models
- Web interface for easy usage
- Real-time sentiment analysis API
- Support for more languages
- Advanced visualization of results
