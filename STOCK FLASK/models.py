import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

def get_financial_data(ticker):
    return yf.Ticker(ticker).history(period="1y")

def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean().iloc[-1]

def calculate_ema(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return rsi

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def calculate_market_cap(ticker):
    stock = yf.Ticker(ticker)
    market_price = stock.history(period='1d')['Close'].iloc[-1]
    shares_outstanding = stock.info['sharesOutstanding']
    market_cap = market_price * shares_outstanding
    return market_cap

def predict_stock_price(data, days_ahead):
    data.reset_index(inplace=True, drop=False)
    data['Days'] = data.index
    X = np.array(data['Days']).reshape(-1, 1)
    y = np.array(data['Close'])
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.array([len(data) + i for i in range(days_ahead)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions.tolist()