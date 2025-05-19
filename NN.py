import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Download data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT']
data = yf.download(tickers, start="2015-01-01", end="2024-01-01", auto_adjust=True)['Close']
data = data.dropna()

# Step 2: Visualize raw daily prices
data.plot(figsize=(12, 6), title="Daily Adjusted Close Prices (2015–2024)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Calculate daily returns
daily_returns = data.pct_change().dropna()

# Step 4: Resample to weekly average daily returns
weekly_avg_returns = daily_returns.resample('W').mean()

# Step 5: Plot weekly average daily returns
weekly_avg_returns.plot(figsize=(12, 6), title="Weekly Average Daily Returns")
plt.xlabel("Week")
plt.ylabel("Average Return")
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = StandardScaler()
normalized_returns = pd.DataFrame(
    scaler.fit_transform(weekly_avg_returns),
    index=weekly_avg_returns.index,
    columns=weekly_avg_returns.columns
)

normalized_returns.plot(figsize=(12, 6), title="Normalized Weekly Average Daily Returns")
plt.xlabel("Week")
plt.ylabel("Normalized Return")
plt.grid(True)
plt.tight_layout()
plt.show()