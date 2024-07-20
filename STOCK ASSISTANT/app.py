from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def get_financial_data(ticker):
    return yf.Ticker(ticker).history(period="1y")

@app.route('/get_stock_price', methods=['GET'])
def get_stock_price():
    ticker = request.args.get('ticker')
    if ticker:
        price = get_financial_data(ticker)['Close'].iloc[-1]
        return jsonify({'ticker': ticker, 'latest_price': price})
    return jsonify({'error': 'Ticker parameter is missing'}), 400

@app.route('/calculate_sma', methods=['GET'])
def calculate_sma():
    ticker = request.args.get('ticker')
    window = int(request.args.get('window', 20))
    if ticker:
        data = get_financial_data(ticker)
        sma = data['Close'].rolling(window=window).mean().iloc[-1]
        return jsonify({'ticker': ticker, 'SMA': sma})
    return jsonify({'error': 'Ticker or window parameter is missing'}), 400

@app.route('/calculate_ema', methods=['GET'])
def calculate_ema():
    ticker = request.args.get('ticker')
    window = int(request.args.get('window', 20))
    if ticker:
        data = get_financial_data(ticker)
        ema = data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
        return jsonify({'ticker': ticker, 'EMA': ema})
    return jsonify({'error': 'Ticker or window parameter is missing'}), 400

@app.route('/calculate_rsi', methods=['GET'])
def calculate_rsi():
    ticker = request.args.get('ticker')
    if ticker:
        data = get_financial_data(ticker)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=14-1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=14-1, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        return jsonify({'ticker': ticker, 'RSI': rsi})
    return jsonify({'error': 'Ticker parameter is missing'}), 400

@app.route('/calculate_macd', methods=['GET'])
def calculate_macd():
    ticker = request.args.get('ticker')
    if ticker:
        data = get_financial_data(ticker)
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return jsonify({'ticker': ticker, 'MACD': macd.iloc[-1], 'Signal Line': signal.iloc[-1]})
    return jsonify({'error': 'Ticker parameter is missing'}), 400

@app.route('/calculate_market_cap', methods=['GET'])
def calculate_market_cap():
    ticker = request.args.get('ticker')
    if ticker:
        stock = yf.Ticker(ticker)
        market_price = stock.history(period='1d')['Close'].iloc[-1]
        shares_outstanding = stock.info['sharesOutstanding']
        market_cap = market_price * shares_outstanding
        return jsonify({'ticker': ticker, 'Market Cap': market_cap})
    return jsonify({'error': 'Ticker parameter is missing'}), 400

@app.route('/predict_price', methods=['GET'])
def predict_price():
    ticker = request.args.get('ticker')
    days_ahead = int(request.args.get('days_ahead'))
    if ticker and days_ahead:
        data = yf.Ticker(ticker).history(period='5y')
        data.reset_index(inplace=True, drop=False)
        data['Days'] = data.index
        X = np.array(data['Days']).reshape(-1, 1)
        y = np.array(data['Close'])
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.array([len(data) + i for i in range(days_ahead)]).reshape(-1, 1)
        predictions = model.predict(future_days)
        return jsonify({'ticker': ticker, 'predictions': predictions.tolist()})
    return jsonify({'error': 'Ticker or days ahead parameter is missing'}), 400

if __name__ == '__main__':
    app.run(debug=True)