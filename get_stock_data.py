import yfinance as yf
import numpy as np

# Download historical stock data
tickers = ['AAPL', 'GOOGL']

for ticker in tickers:
    stock_data = yf.download(ticker, start='2020-01-01', end='2023-09-27')

    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Close'].pct_change()

    # Calculate the average return (expected return)
    expected_return = stock_data['Daily Return'].mean()

    # Convert to annualized expected return (for daily returns, multiply by 252 trading days)
    annualized_return = (1 + expected_return) ** 252 - 1

    print(f"Expected Daily Return: {expected_return:.4%}")
    print(f"Expected Annualized Return: {annualized_return:.4%}")
