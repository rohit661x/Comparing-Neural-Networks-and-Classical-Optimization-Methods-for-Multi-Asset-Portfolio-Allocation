import yfinance as yf
import pandas as pd

tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT']
data = yf.download(tickers, start="2015-01-01", end="2024-01-01", auto_adjust=True)['Close']
data = data.dropna()
print(data.head())
