# Sentiment Analysis with Python: NLTK VADER, RoBERTa, and Transformers

*Download the dataset 'Review.csv' using the below link*

*https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/input?select=Reviews.csv*

## VADER (Valence Aware Dictionary for sEntiment Reasoning)
- **Overview**: An NLP algorithm that uses a sentiment lexicon-based approach combined with grammatical rules and syntactical conventions to return the polarity of a sentence (positive or negative).
- **Score Range**: From -4 (most negative) to 4 (most positive).
- **Sentiment Lexicon**: 
  - Handles case sensitivity (e.g., "very" vs. "VERY"), emoticons (😊), online slangs (e.g., "da bomb"), and abbreviations (e.g., "ROFL").
- **Grammatical Rules**:
  - Incorporates heuristic rules for punctuation, capitalization, adverbs, and contrastive conjunctions.
- **Polarity Scores**:
  - Negative
  - Neutral
  - Positive
  - Compound (combined overall sentiment)
- **Implementation**:
  - NLTK’s `SentimentIntensityAnalyzer` is used to compute the polarity scores in a dictionary format.
  - Uses a "bag of words" approach:
    - Stop words are removed.
    - Each word is individually scored and combined into a total score.
### Pros
  - No need for text preprocessing.
  - Built-in stop words removal.
### Cons
  - Words are scored individually, without considering relationships between them.
  - Doesn't account for the overall context of the sentence.
  - Ineffective in detecting sarcasm.

## RoBERTa (Robustly Optimized BERT)
- **Overview**: A language model and a variant of BERT (Bidirectional Encoder Representations from Transformers) optimized for better performance in NLP tasks.
- **Type**: Transformer-based deep learning pretrained model.
- **Usage**:
  - Tokenization via `AutoTokenizer.from_pretrained(model_name)` from the Huggingface `Transformers` library.
  - Model loading via `AutoModelForSequenceClassification.from_pretrained(model_name)`.
### Pros
  - Considers both words and the context of surrounding words.
  - Offers more accurate predictions compared to VADER.
- **Implementation**: Works seamlessly with the Huggingface `Transformers` library for advanced NLP tasks.

## Transformer Pipelines
- **Overview**: An easy-to-use utility in the Huggingface `Transformers` library for various NLP tasks.
- **Module**: `pipeline` (e.g., for sentiment analysis).
### Pros
  - Simple and easy to implement.
  - Suitable for smaller datasets.
### Cons
  - Slow performance on large datasets.
  - Struggles with predicting sentiment on large quantities of text.
