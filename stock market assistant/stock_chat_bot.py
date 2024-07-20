from flask import Flask, request, jsonify
import yfinance as yf
from newsapi import NewsApiClient 
import openai
import nltk
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')

app = Flask(__name__)

# Initialize News API Client
newsapi = NewsApiClient(api_key='483cbf9128594c58a43b38085bd768f5')

# OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# A sample mapping of company names to ticker symbols
company_to_ticker = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'amazon': 'AMZN',
    'facebook': 'META',
    'tesla': 'TSLA',
    'netflix': 'NFLX'
}

@app.route('/api/stock-price/<ticker>', methods=['GET'])
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period='1d')['Close'].iloc[-1]
    return jsonify({'price': price})

@app.route('/api/sma/<ticker>/<int:window>', methods=['GET'])
def calculate_sma(ticker, window):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')['Close']
    sma = data.rolling(window=window).mean().iloc[-1]
    return jsonify({'sma': sma})

@app.route('/api/ema/<ticker>/<int:window>', methods=['GET'])
def calculate_ema(ticker, window):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')['Close']
    ema = data.ewm(span=window, adjust=False).mean().iloc[-1]
    return jsonify({'ema': ema})

@app.route('/api/rsi/<ticker>', methods=['GET'])
def calculate_rsi(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')['Close']
    delta = data.diff()
    gain = delta.astype(float).where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return jsonify({'rsi': rsi.iloc[-1]})

@app.route('/api/bollinger-bands/<ticker>', methods=['GET'])
def calculate_bollinger_bands(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')['Close']
    sma = data.rolling(window=20).mean()
    std = data.rolling(window=20).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return jsonify({'upper_band': upper_band.iloc[-1], 'lower_band': lower_band.iloc[-1]})

@app.route('/api/pe-ratio/<ticker>', methods=['GET'])
def get_pe_ratio(ticker):
    stock = yf.Ticker(ticker)
    pe_ratio = stock.info['forwardPE']
    return jsonify({'pe_ratio': pe_ratio})

@app.route('/api/dividend-yield/<ticker>', methods=['GET'])
def get_dividend_yield(ticker):
    stock = yf.Ticker(ticker)
    dividend_yield = stock.info['dividendYield'] * 100
    return jsonify({'dividend_yield': dividend_yield})

@app.route('/api/volume/<ticker>', methods=['GET'])
def get_volume(ticker):
    stock = yf.Ticker(ticker)
    volume = stock.history(period='1d')['Volume'].iloc[-1]
    return jsonify({'volume': volume})

@app.route('/api/market-cap/<ticker>', methods=['GET'])
def calculate_market_cap(ticker):
    stock = yf.Ticker(ticker)
    market_cap = stock.info['marketCap']
    return jsonify({'market_cap': market_cap})

@app.route('/api/news', methods=['GET'])
def get_news():
    news = newsapi.get_top_headlines(category='business', language='en', country='us')
    articles = news['articles']
    for article in articles:
        analysis = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the sentiment of this news article and provide a recommendation (buy, sell, hold): {article['title']} {article['description']}",
            max_tokens=50
        )
        article['sentiment'] = analysis.choices[0].text.strip()
    return jsonify(articles)

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    message = data.get('message') if data is not None else None
    tokens = word_tokenize(message.lower())
    ticker = extract_ticker_symbol_or_name(tokens)
    if not ticker:
        return jsonify({'response': "Could not identify the ticker symbol."}), 400
    
    # Determine the request type and respond accordingly
    if 'price' in tokens:
        return get_stock_price(ticker)
    elif 'sma' in tokens or 'simple moving average' in tokens:
        for token in tokens:
            if token.isdigit():
                window = int(token)
                return calculate_sma(ticker, window)
    elif 'ema' in tokens or 'exponential moving average' in tokens:
        for token in tokens:
            if token.isdigit():
                window = int(token)
                return calculate_ema(ticker, window)
    elif 'rsi' in tokens or 'relative strength index' in tokens:
        return calculate_rsi(ticker)
    elif 'bollinger bands' in tokens:
        return calculate_bollinger_bands(ticker)
    elif 'p/e ratio' in tokens or 'price-to-earnings ratio' in tokens:
        return get_pe_ratio(ticker)
    elif 'dividend yield' in tokens:
        return get_dividend_yield(ticker)
    elif 'volume' in tokens:
        return get_volume(ticker)
    elif 'market cap' in tokens or 'market capitalization' in tokens:
        return calculate_market_cap(ticker)
    else:
        return jsonify({'response': "Invalid query."}), 400
    
def extract_ticker_symbol_or_name(tokens):
    for token in tokens:
        if token in company_to_ticker:
            return company_to_ticker[token]
        elif token.isalpha() and len(token) <= 5:
            return token.upper()
    return None

if __name__ == "__main__":
    app.run(debug=True)