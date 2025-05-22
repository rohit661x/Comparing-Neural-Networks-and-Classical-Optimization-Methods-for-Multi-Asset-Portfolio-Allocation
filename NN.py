import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

# Normalize weekly returns (for visualization or NN input)
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


# === New Step: Mean-Variance Optimization (MVO) ===

def mean_variance_optimization(returns_df, target_return=None):
    """
    Perform Markowitz Mean-Variance Optimization on the returns DataFrame.
    
    returns_df: pd.DataFrame of returns (weekly average returns)
    target_return: float target portfolio return; if None, use mean of asset expected returns
    
    Returns: optimal weights as numpy array
    """
    mu = returns_df.mean().values          # Expected returns vector
    Sigma = returns_df.cov().values         # Covariance matrix
    n = len(mu)

    if target_return is None:
        target_return = mu.mean()
    
    # Define optimization variable
    w = cp.Variable(n)
    
    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    
    # Constraints: weights sum to 1, no short selling, expected return >= target return
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        mu @ w >= target_return
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception(f"Solver failed with status {problem.status}")
    
    return w.value

# Run MVO on weekly average returns
optimal_weights = mean_variance_optimization(weekly_avg_returns)

print("Optimal portfolio weights from MVO:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

# --- Efficient Frontier Visualization ---

def efficient_frontier(returns_df, points=50):
    """
    Compute efficient frontier for a range of target returns.
    Returns arrays of risks (std dev) and returns.
    """
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)

    # Range of target returns from min to max mean return of assets
    target_returns = np.linspace(mu.min(), mu.max(), points)
    risks = []
    rets = []

    for r_target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, w >= 0, mu @ w >= r_target]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status in ["optimal", "optimal_inaccurate"]:
            risks.append(np.sqrt(w.value.T @ Sigma @ w.value))
            rets.append(mu @ w.value)
        else:
            # If no solution, append NaN
            risks.append(np.nan)
            rets.append(np.nan)

    return np.array(risks), np.array(rets)

# Compute frontier
risks, rets = efficient_frontier(weekly_avg_returns)

# Calculate individual assets’ risk-return points
asset_means = weekly_avg_returns.mean()
asset_stds = weekly_avg_returns.std()

plt.figure(figsize=(12, 7))
plt.plot(risks, rets, 'b-', label='Efficient Frontier')
plt.scatter(asset_stds, asset_means, c='red', marker='o', label='Individual Assets')

# Highlight the MVO optimal portfolio point
opt_risk = np.sqrt(optimal_weights.T @ weekly_avg_returns.cov().values @ optimal_weights)
opt_return = optimal_weights @ weekly_avg_returns.mean().values
plt.scatter(opt_risk, opt_return, c='green', marker='*', s=200, label='Optimal MVO Portfolio')

plt.title("Efficient Frontier and Individual Asset Risk-Return")
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


window_size = 52  # 52 weeks ≈ 1 year rolling window
rolling_weights = []

for i in range(window_size, len(weekly_avg_returns) + 1):
    window_returns = weekly_avg_returns.iloc[i - window_size:i]
    try:
        w = mean_variance_optimization(window_returns)
        rolling_weights.append(w)
    except Exception:
        rolling_weights.append(np.full(len(tickers), np.nan))

rolling_weights = pd.DataFrame(
    rolling_weights,
    index=weekly_avg_returns.index[window_size - 1:],
    columns=tickers
)

print(rolling_weights.tail())

# Calculate weekly portfolio returns using optimal MVO weights
portfolio_weekly_returns = weekly_avg_returns @ optimal_weights

# Calculate cumulative returns for portfolio and each asset
cumulative_portfolio_returns = (1 + portfolio_weekly_returns).cumprod() - 1
cumulative_asset_returns = (1 + weekly_avg_returns).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(12, 7))
plt.plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns, label='MVO Portfolio', linewidth=3, color='black')

for ticker in weekly_avg_returns.columns:
    plt.plot(cumulative_asset_returns.index, cumulative_asset_returns[ticker], label=ticker)

plt.title("Cumulative Returns: MVO Portfolio vs Individual Assets (2015-2024)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def create_features(returns_df, window_size=12):
    features = []
    targets = []

    for i in range(len(returns_df) - window_size):
        window = returns_df.iloc[i:i+window_size].values.flatten()
        next_week = returns_df.iloc[i+window_size].values
        features.append(window)
        targets.append(next_week / next_week.sum())  # convert to proportional weights as "ground truth" (heuristic)

    return np.array(features), np.array(targets)

X, y = create_features(normalized_returns, window_size=12)

class PortfolioNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PortfolioNN, self).__init__()
        # input_size will be lookback_period * num_assets
        # output_size will be num_assets (the portfolio weights)
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        # Output layer for weights. Softmax ensures they sum to 1 and are positive.
        self.fc3 = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1) # Apply softmax across the asset dimension

    def forward(self, x):
        # Flatten the input from (batch_size, lookback_period, num_assets)
        # to (batch_size, lookback_period * num_assets) for the linear layer
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PortfolioNN(input_dim=X_train.shape[1], output_size=y_train.shape[1])
model = model.to(device)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

loss_history = []

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")




plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()