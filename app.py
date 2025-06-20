import streamlit as st
import requests
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# --- Section 3.2.2: Fetching Live Financial News ---
def fetch_news(ticker):
    api_key = "b565b5757ecd4460ae5084dbcfc2371f"  # Replace with your actual NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# --- Section 3.3.1: Sentiment Analysis Using FinBERT ---
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        title = article['title']
        sentiment_result = sentiment_pipeline(title)[0]
        sentiments.append({
            'title': title,
            'sentiment': sentiment_result['label']
        })
    return sentiments

# --- Section 3.4.1: Rule-Based Prediction Logic ---
def predict_stock_movement(sentiments):
    positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
    negative_count = len(sentiments) - positive_count
    if positive_count > negative_count:
        return "Prediction: Stock is likely to go UP!"
    else:
        return "Prediction: Stock is likely to go DOWN!"

# --- Section 3.5.1: Streamlit Interface ---
st.title("Stock Sentiment Analysis App")

tickers = ['TSLA', 'AMZN']
selected_ticker = st.selectbox("Select a Stock Ticker", tickers)

# --- Fetch and analyze on selection ---
if selected_ticker:
    articles = fetch_news(selected_ticker)
    sentiments = analyze_sentiment(articles)

    # --- Section 3.5.2: Displaying Sentiment Results ---
    def display_sentiment_scores(sentiments):
        st.write("### Sentiment Analysis Results:")
        for sentiment in sentiments:
            st.write(f"**Title:** {sentiment['title']}")
            st.write(f"**Sentiment:** {sentiment['sentiment']}")
            st.write("---")

    display_sentiment_scores(sentiments)

    # --- Section 3.5.3: Visualizing Sentiment Distribution (Updated for future-proofing) ---
    def plot_sentiment(sentiments):
        sentiment_counts = pd.Series([s['sentiment'] for s in sentiments]).value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
        ax.set_title('Sentiment Analysis of Stock News')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    plot_sentiment(sentiments)

    # --- Section 3.4.1: Prediction Output ---
    prediction = predict_stock_movement(sentiments)
    st.subheader(prediction)

    # --- Section 3.5.4: Visualizing Stock Prices (placeholder function) ---
    def fetch_stock_data(ticker):
        # Placeholder: Replace with actual API like yfinance or Alpha Vantage
        # Example using dummy data:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=7)
        prices = [150 + i * 2 for i in range(7)]
        return pd.DataFrame({'Date': dates, 'Close': prices}).set_index('Date')

    stock_data = fetch_stock_data(selected_ticker)
    st.line_chart(stock_data['Close'])
