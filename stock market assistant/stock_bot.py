import yfinance as yf
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Function to get stock price
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period='1d')['Close'].iloc[-1]
        return f"The current price of {ticker} is ${price:.2f}"
    except Exception as e:
        return f"Could not retrieve stock price for {ticker}. Error: {e}"

# Function to calculate Simple Moving Average (SMA)
def calculate_SMA(ticker, window):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')['Close']
        sma = data.rolling(window=window).mean().iloc[-1]
        return f"The {window}-day SMA of {ticker} is ${sma:.2f}"
    except Exception as e:
        return f"Could not calculate SMA for {ticker}. Error: {e}"

# Function to calculate Exponential Moving Average (EMA)
def calculate_EMA(ticker, window):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')['Close']
        ema = data.ewm(span=window, adjust=False).mean().iloc[-1]
        return f"The {window}-day EMA of {ticker} is ${ema:.2f}"
    except Exception as e:
        return f"Could not calculate EMA for {ticker}. Error: {e}"

# Function to calculate Relative Strength Index (RSI)
def calculate_RSI(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')['Close']
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return f"The RSI of {ticker} is {rsi.iloc[-1]:.2f}"
    except Exception as e:
        return f"Could not calculate RSI for {ticker}. Error: {e}"

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')['Close']
        sma = data.rolling(window=20).mean()
        std = data.rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return (f"The upper Bollinger Band of {ticker} is ${upper_band.iloc[-1]:.2f}, "
                f"the lower Bollinger Band is ${lower_band.iloc[-1]:.2f}")
    except Exception as e:
        return f"Could not calculate Bollinger Bands for {ticker}. Error: {e}"

# Function to calculate Price-to-Earnings (P/E) Ratio
def get_PE_ratio(ticker):
    try:
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info['forwardPE']
        return f"The P/E ratio of {ticker} is {pe_ratio:.2f}"
    except Exception as e:
        return f"Could not retrieve P/E ratio for {ticker}. Error: {e}"

# Function to calculate Dividend Yield
def get_dividend_yield(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividend_yield = stock.info['dividendYield'] * 100
        return f"The dividend yield of {ticker} is {dividend_yield:.2f}%"
    except Exception as e:
        return f"Could not retrieve dividend yield for {ticker}. Error: {e}"

# Function to get trading volume
def get_volume(ticker):
    try:
        stock = yf.Ticker(ticker)
        volume = stock.history(period='1d')['Volume'].iloc[-1]
        return f"The trading volume of {ticker} is {volume}"
    except Exception as e:
        return f"Could not retrieve trading volume for {ticker}. Error: {e}"

# Function to calculate Market Capitalization
def calculate_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info['marketCap']
        return f"The market capitalization of {ticker} is ${market_cap:,}"
    except Exception as e:
        return f"Could not retrieve market capitalization for {ticker}. Error: {e}"

# Function to extract ticker symbol from message
def extract_ticker_symbol(tokens):
    for token in tokens:
        if token.isalpha() and len(token) <= 5:
            return token.upper()
    return None

# Function to process user input
def process_message(message):
    tokens = word_tokenize(message.lower())
    ticker = extract_ticker_symbol(tokens)
    if not ticker:
        return "Sorry, I couldn't identify the ticker symbol. Please provide a valid ticker symbol."
    
    if 'price' in tokens:
        return get_stock_price(ticker)
    elif 'sma' in tokens or 'simple moving average' in tokens:
        for i, token in enumerate(tokens):
            if token.isdigit():
                window = int(token)
                return calculate_SMA(ticker, window)
    elif 'ema' in tokens or 'exponential moving average' in tokens:
        for i, token in enumerate(tokens):
            if token.isdigit():
                window = int(token)
                return calculate_EMA(ticker, window)
    elif 'rsi' in tokens or 'relative strength index' in tokens:
        return calculate_RSI(ticker)
    elif 'bollinger bands' in tokens:
        return calculate_bollinger_bands(ticker)
    elif 'p/e ratio' in tokens or 'price-to-earnings ratio' in tokens:
        return get_PE_ratio(ticker)
    elif 'dividend yield' in tokens:
        return get_dividend_yield(ticker)
    elif 'volume' in tokens:
        return get_volume(ticker)
    elif 'market cap' in tokens or 'market capitalization' in tokens:
        return calculate_market_cap(ticker)
    else:
        return "Sorry, I didn't understand your request. Please ask about stock price, SMA, EMA, RSI, Bollinger Bands, P/E ratio, dividend yield, volume, or market cap."

# Main function to run the chatbot
def main():
    print("Welcome to the Stock Market Chatbot!")
    print("You can ask about stock prices, SMAs, EMAs, RSI, Bollinger Bands, P/E ratios, dividend yields, volumes, and market capitalization.")
    print("Examples:")
    print("- 'What is the price of AAPL?'")
    print("- 'Calculate SMA for MSFT 20'")
    print("- 'Show market cap for TSLA'")
    print("- 'What is the EMA of GOOGL 50?'")
    print("- 'Give me the RSI of AMZN'")
    print("- 'Calculate Bollinger Bands for FB'")
    print("- 'What is the P/E ratio of NFLX?'")
    print("- 'What is the dividend yield of KO?'")
    print("- 'What is the volume of NVDA?'")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = process_message(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()