# Twitter Sentiment Analysis Project

## Overview
This project aims to perform sentiment analysis on Twitter data using machine learning techniques. The goal is to classify tweets as either positive or negative based on their content.

## Dataset
The dataset used in this project is sourced from Twitter and contains tweets labeled with sentiment. It consists of various features including tweet text, user information, and timestamps. The target variable indicates whether the tweet is positive or negative.

## Preprocessing
- **Cleaning**: Text data is preprocessed to remove noise and irrelevant information.
- **Stemming**: Words are stemmed using the Porter stemming algorithm to reduce them to their root form.
- **Vectorization**: Text data is transformed into numerical vectors using TF-IDF vectorization.

## Machine Learning Model
- **Model**: Logistic Regression is utilized as the classification algorithm.
- **Training**: The model is trained on the preprocessed tweet data.
- **Evaluation**: Model accuracy is evaluated on both training and testing datasets.

## Results
The trained model achieves a high accuracy score on both training and testing data, indicating its effectiveness in classifying tweet sentiments.

## Saving Model
The trained model along with the TF-IDF vectorizer is saved using the `pickle` library for future use.

## Usage
To predict sentiment for a given text, use the `predict_sentiment()` function provided in the `sentiment_analysis.py` script.

```python
from sentiment_analysis import predict_sentiment

# Example usage
text = "I love puppies"
sentiment = predict_sentiment(text)
print(sentiment)  # Output: "Positive sentiment"
```

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
