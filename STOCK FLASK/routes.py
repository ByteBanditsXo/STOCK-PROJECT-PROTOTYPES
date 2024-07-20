from flask import request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from models import (
    get_financial_data, calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_market_cap, predict_stock_price
)
from scraper import scrape_data  # Import the scrape_data function

def configure_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/fetch_news', methods=['GET'])
    def fetch_news():
        ticker = request.args.get('ticker')
        if ticker:
            news = scrape_data(ticker)
            return jsonify(news)
        return jsonify({'error': 'Ticker is required'}), 400

    @app.route('/get_data', methods=['GET'])
    def get_data():
        ticker = request.args.get('ticker')
        if ticker:
            data = get_financial_data(ticker)
            sma = calculate_sma(data)
            ema = calculate_ema(data)
            rsi = calculate_rsi(data)
            macd, signal = calculate_macd(data)
            market_cap = calculate_market_cap(ticker)
            predictions = predict_stock_price(data, 30)
            return jsonify({
                'SMA': sma,
                'EMA': ema,
                'RSI': rsi,
                'MACD': macd,
                'Signal Line': signal,
                'Market Cap': market_cap,
                'Predictions': predictions
            })
        return jsonify({'error': 'Ticker is required'}), 400