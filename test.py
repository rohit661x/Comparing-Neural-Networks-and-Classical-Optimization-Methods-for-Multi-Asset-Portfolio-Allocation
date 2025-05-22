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

# --- Configuration ---
# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Collection and Preprocessing ---
def get_and_preprocess_data(tickers, start_date, end_date):
    """
    Downloads data, calculates weekly average daily returns, and normalizes them.
    """
    print("Step 1: Downloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    data = data.dropna(axis=0, how='any') # Drop rows with any NaN (e.g., missing data for a ticker)
    data = data.dropna(axis=1, how='all') # Drop columns (tickers) that are entirely NaN

    if data.empty:
        raise ValueError("No valid data retrieved after dropping NaNs. Check tickers/dates.")

    print("Step 2: Calculating daily returns...")
    daily_returns = data.pct_change().dropna()

    print("Step 3: Resampling to weekly average daily returns...")
    # Using .mean() for weekly average daily returns as per your code
    weekly_avg_returns = daily_returns.resample('W').mean().dropna()

    if weekly_avg_returns.empty:
        raise ValueError("No weekly average returns generated. Check data range.")

    print("Step 4: Normalizing weekly returns...")
    scaler = StandardScaler()
    normalized_returns = pd.DataFrame(
        scaler.fit_transform(weekly_avg_returns),
        index=weekly_avg_returns.index,
        columns=weekly_avg_returns.columns
    )
    return data, weekly_avg_returns, normalized_returns, scaler

# --- MVO Function for NN Target Generation (Maximize Return for Given Risk) ---
# Using your provided mean_variance_optimization function
def mean_variance_optimization(returns_df_window, target_return=None):
    """
    Perform Markowitz Mean-Variance Optimization on a window of returns data.
    This will be used to generate 'y' (target weights) for the neural network.

    returns_df_window: pd.DataFrame of returns (e.g., weekly average returns for a window)
    target_return: float target portfolio return; if None, use mean of asset expected returns
                   If None, it calculates the Global Minimum Variance (GMV) portfolio.
    Returns: optimal weights as numpy array, or None if optimization fails.
    """
    mu = returns_df_window.mean().values      # Expected returns vector for this window
    Sigma = returns_df_window.cov().values    # Covariance matrix for this window
    n = len(mu)

    # Handle potential singular covariance matrix (e.g., if window_size is too small or assets are perfectly correlated)
    # Add a small diagonal to ensure positive definiteness
    Sigma += np.eye(n) * 1e-7

    # Define optimization variable
    w = cp.Variable(n)

    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Constraints: weights sum to 1, no short selling
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
    ]

    # Add target return constraint ONLY if target_return is specified
    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(verbose=False) # Suppress solver output
        if problem.status in ["optimal", "optimal_inaccurate"]:
            # Ensure weights are non-negative and sum to 1 due to potential solver inaccuracies
            weights = w.value
            weights[weights < 0] = 0 # Clip negative weights
            if np.sum(weights) > 0: # Avoid division by zero
                weights /= np.sum(weights) # Re-normalize to sum to 1
            else: # If all weights are effectively zero, return equal weights or handle as failure
                return np.full(n, 1/n) # Fallback to equal weights
            return weights
        else:
            # print(f"MVO Solver failed with status {problem.status} for window ending {returns_df_window.index[-1]}")
            return None # Optimization failed
    except Exception as e: # Catch general exceptions from solver
        # print(f"MVO Solver error: {e} for window ending {returns_df_window.index[-1]}")
        return None

# --- Neural Network Model (PortfolioNN) ---
class PortfolioNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PortfolioNN, self).__init__()
        # input_size will be lookback_period * num_assets
        # output_size will be num_assets (the portfolio weights)
        self.fc1 = nn.Linear(input_size, 128) # Increased hidden layer size
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)      # Increased Dropout rate
        self.fc2 = nn.Linear(128, 64)        # Increased hidden layer size
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)      # Increased Dropout rate
        # Output layer for weights. Softmax ensures they sum to 1 and are positive.
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1) # Apply softmax across the asset dimension

    def forward(self, x):
        # Flatten the input from (batch_size, lookback_period, num_assets)
        # to (batch_size, lookback_period * num_assets) for the linear layer
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x) # Apply dropout
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x) # Apply dropout
        x = self.softmax(self.fc3(x))
        return x

# --- Data Preparation for NN Training ---
def prepare_nn_data_for_portfolio_training(normalized_returns_df, lookback_period=52, mvo_target_return_for_nn_y=None, validation_split=0.1):
    """
    Prepares input sequences (historical normalized returns) and target outputs (MVO weights)
    for the neural network. Includes a validation split for early stopping.
    """
    print(f"\nPreparing NN data with lookback_period={lookback_period}...")
    num_assets = normalized_returns_df.shape[1]
    sequences = [] # X data: flattened historical normalized returns
    targets = []   # y data: MVO optimal weights

    # Iterate to create sequences and corresponding MVO targets
    for i in range(lookback_period, len(normalized_returns_df)):
        # Input sequence: historical normalized returns for 'lookback_period'
        history_df_window = normalized_returns_df.iloc[i - lookback_period:i]

        # Target: MVO weights calculated based on this 'history_df_window'
        # Pass the target_return to the MVO function
        optimal_weights = mean_variance_optimization(history_df_window, target_return=mvo_target_return_for_nn_y)

        if optimal_weights is not None:
            # X: Flatten the history array for input to the NN
            sequences.append(history_df_window.values.flatten())
            # Y: Append the optimal weights calculated by MVO
            targets.append(optimal_weights)

    if not sequences:
        raise ValueError("No data sequences generated for NN. Adjust lookback period or data range.")

    # Convert to NumPy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # Split into training, validation, and testing sets (shuffle=False for time series data)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_split, shuffle=False, random_state=42)


    # Convert to PyTorch Tensors (FloatTensor for both X and y, as y are continuous weights)
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device) # y is now float (weights)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)   # y is now float (weights)

    # Create DataLoaders for efficient batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # No shuffle for validation
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"  Total NN samples: {len(sequences)}")
    print(f"  Training samples: {len(X_train_tensor)}")
    print(f"  Validation samples: {len(X_val_tensor)}")
    print(f"  Testing samples: {len(X_test_tensor)}")

    return train_loader, val_loader, test_loader, X_test_tensor, y_test_tensor, num_assets, lookback_period

# --- Plotting Function ---
def plot_training_loss(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs.
    """
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, marker='o', linestyle='-', color='skyblue', label='Training Loss')
    plt.plot(epochs_range, val_losses, marker='x', linestyle='--', color='salmon', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Portfolio Evaluation Metrics ---
def evaluate_portfolio_performance(returns_series_or_df, weights):
    """
    Evaluates the performance of a portfolio given its weights and asset returns.
    Assumes returns_series_or_df contains the actual (unnormalized) returns for the period.
    """
    if weights is None or len(weights) == 0 or np.sum(weights) == 0:
        return {'annualized_return': 0, 'annualized_volatility': 0, 'sharpe_ratio': 0, 'cumulative_return': 0}

    # Ensure weights are normalized to sum to 1 for correct portfolio calculation
    weights = np.array(weights)
    weights[weights < 0] = 0 # Clip any negative weights if they somehow appear
    if np.sum(weights) == 0: # Avoid division by zero if all weights are zero
        return {'annualized_return': 0, 'annualized_volatility': 0, 'sharpe_ratio': 0, 'cumulative_return': 0}
    weights = weights / np.sum(weights) # Re-normalize

    portfolio_returns = returns_series_or_df.dot(weights)
    annualized_return = portfolio_returns.mean() * 52 # Assuming weekly returns
    annualized_volatility = portfolio_returns.std() * np.sqrt(52) # Assuming weekly returns
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    cumulative_return = (1 + portfolio_returns).prod() - 1

    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return
    }

# --- Main Execution ---
if __name__ == '__main__':
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT']
    start_date = "2015-01-01"
    end_date = "2024-01-01"
    lookback_period = 12 # 52 weeks = 1 year rolling window for NN input
    # Target return for MVO calculation (for the NN's 'y' targets)
    # Set to None to target the Global Minimum Variance (GMV) portfolio, which is often more stable.
    mvo_target_return_for_nn_y = None # Changed to None for GMV target

    try:
        # 1. Data Collection and Preprocessing
        raw_prices, weekly_avg_returns, normalized_returns, scaler = \
            get_and_preprocess_data(tickers, start_date, end_date)

        # Visualize raw daily prices
        raw_prices.plot(figsize=(12, 6), title="Daily Adjusted Close Prices (2015–2024)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot weekly average daily returns
        weekly_avg_returns.plot(figsize=(12, 6), title="Weekly Average Daily Returns")
        plt.xlabel("Week")
        plt.ylabel("Average Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot normalized weekly average daily returns
        normalized_returns.plot(figsize=(12, 6), title="Normalized Weekly Average Daily Returns")
        plt.xlabel("Week")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- MVO on Full Period (for comparison/visualization) ---
        print("\n--- MVO on Full Period ---")
        # For full period MVO, let's target the mean of asset returns for a balanced view
        full_period_mvo_weights = mean_variance_optimization(weekly_avg_returns, target_return=weekly_avg_returns.mean().mean())
        if full_period_mvo_weights is not None:
            print("Optimal portfolio weights from MVO (Full Period):")
            for ticker, weight in zip(tickers, full_period_mvo_weights):
                print(f"{ticker}: {weight:.4f}")
        else:
            print("MVO on full period failed.")

        # --- Efficient Frontier Visualization (Full Period) ---
        def efficient_frontier(returns_df, points=50):
            mu = returns_df.mean().values
            Sigma = returns_df.cov().values
            n = len(mu)
            target_returns = np.linspace(mu.min(), mu.max(), points)
            risks = []
            rets = []
            for r_target in target_returns:
                w = cp.Variable(n)
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                constraints = [cp.sum(w) == 1, w >= 0, mu @ w >= r_target]
                problem = cp.Problem(objective, constraints)
                problem.solve(verbose=False)
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    risks.append(np.sqrt(w.value.T @ Sigma @ w.value))
                    rets.append(mu @ w.value)
                else:
                    risks.append(np.nan)
                    rets.append(np.nan)
            return np.array(risks), np.array(rets)

        risks, rets = efficient_frontier(weekly_avg_returns)
        asset_means = weekly_avg_returns.mean()
        asset_stds = weekly_avg_returns.std()

        plt.figure(figsize=(12, 7))
        plt.plot(risks, rets, 'b-', label='Efficient Frontier')
        plt.scatter(asset_stds, asset_means, c='red', marker='o', label='Individual Assets')
        if full_period_mvo_weights is not None:
            opt_risk = np.sqrt(full_period_mvo_weights.T @ weekly_avg_returns.cov().values @ full_period_mvo_weights)
            opt_return = full_period_mvo_weights @ weekly_avg_returns.mean().values
            plt.scatter(opt_risk, opt_return, c='green', marker='*', s=200, label='Optimal MVO Portfolio (Full Period)')

        plt.title("Efficient Frontier and Individual Asset Risk-Return (Full Period)")
        plt.xlabel("Risk (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Rolling MVO Weights (for context, not directly used by NN training as target) ---
        window_size_rolling_mvo = 12 # Your original rolling window for MVO
        rolling_weights = []
        print(f"\nComputing rolling MVO weights with window_size={window_size_rolling_mvo}...")
        for i in range(window_size_rolling_mvo, len(weekly_avg_returns) + 1):
            window_returns = weekly_avg_returns.iloc[i - window_size_rolling_mvo:i]
            try:
                # Use a consistent target_return for rolling MVO as well, e.g., the average mean return of the window
                w = mean_variance_optimization(window_returns, target_return=window_returns.mean().mean())
                rolling_weights.append(w)
            except Exception:
                rolling_weights.append(np.full(len(tickers), np.nan))

        rolling_weights_df = pd.DataFrame(
            rolling_weights,
            index=weekly_avg_returns.index[window_size_rolling_mvo - 1:],
            columns=tickers
        )
        print("\nRolling MVO Weights (tail):")
        print(rolling_weights_df.tail())

        # Calculate weekly portfolio returns using optimal MVO weights (from full period MVO)
        if full_period_mvo_weights is not None:
            portfolio_weekly_returns_full_mvo = weekly_avg_returns @ full_period_mvo_weights
            cumulative_portfolio_returns_full_mvo = (1 + portfolio_weekly_returns_full_mvo).cumprod() - 1
            cumulative_asset_returns_full_period = (1 + weekly_avg_returns).cumprod() - 1

            plt.figure(figsize=(12, 7))
            plt.plot(cumulative_portfolio_returns_full_mvo.index, cumulative_portfolio_returns_full_mvo, label='MVO Portfolio (Full Period)', linewidth=3, color='black')
            for ticker in weekly_avg_returns.columns:
                plt.plot(cumulative_asset_returns_full_period.index, cumulative_asset_returns_full_period[ticker], label=ticker)
            plt.title("Cumulative Returns: MVO Portfolio (Full Period) vs Individual Assets")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        # 2. Prepare Data for NN Training (generates MVO targets for NN)
        # The `lookback_period` for NN input is 52 weeks (1 year)
        # The `mvo_target_return_for_nn_y` defines the specific MVO portfolio the NN will learn to replicate
        train_loader, val_loader, test_loader, X_test_tensor, y_test_tensor, num_assets, _ = \
            prepare_nn_data_for_portfolio_training(normalized_returns, lookback_period, mvo_target_return_for_nn_y, validation_split=0.15) # Increased validation split slightly

        # Check if enough data for training/testing
        if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print("Not enough data to create training/testing sets. Adjust lookback period or data range.")
            exit()

        # 3. Initialize Neural Network Model
        input_size = lookback_period * num_assets
        output_size = num_assets
        model = PortfolioNN(input_size, output_size).to(device)
        print("\nNeural Network Model Architecture:")
        print(model)

        # 4. Set Criterion (Loss Function) and Optimizer
        criterion = nn.MSELoss() # Correct for regression (predicting continuous weights)
        # Added weight_decay for L2 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # L2 regularization

        # 5. Training Loop with Early Stopping
        epochs = 1000 # Max epochs, but early stopping will likely halt it sooner
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 100 # Number of epochs to wait for improvement
        epochs_no_improve = 0
        print("\nStarting NN training...")
        for epoch in range(epochs):
            model.train() # Set model to training mode
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs) # Forward pass
                loss = criterion(outputs, targets) # Calculate loss
                loss.backward() # Backpropagation
                optimizer.step() # Update weights
                running_train_loss += loss.item() * inputs.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Evaluate on validation set
            model.eval() # Set model to evaluation mode
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_val_loss += loss.item() * inputs.size(0)
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            if (epoch + 1) % 50 == 0: # Print every 50 epochs
                print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                # Optionally save the best model state
                # torch.save(model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
                    break # Exit training loop

        print("\nNN Training complete.")

        # 6. Plot Training and Validation Loss
        plot_training_loss(train_losses, val_losses)

        # 7. Evaluate Model on Test Data
        print("\nEvaluating NN model on test data...")
        model.eval() # Set model to evaluation mode
        test_loss = 0.0
        nn_predicted_weights_list = []
        mvo_target_weights_list = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

                nn_predicted_weights_list.extend(outputs.cpu().numpy())
                mvo_target_weights_list.extend(targets.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        print(f"Neural Network Test Loss (MSE): {test_loss:.6f}")

        # Convert lists to numpy arrays for evaluation
        nn_predicted_weights_array = np.array(nn_predicted_weights_list)
        mvo_target_weights_array = np.array(mvo_target_weights_list)

        # Calculate average weights over the test set for overall performance evaluation
        avg_nn_weights = np.mean(nn_predicted_weights_array, axis=0)
        avg_mvo_weights = np.mean(mvo_target_weights_array, axis=0) # This is the average of the MVO targets

        # Determine the actual returns for the test period
        # This requires careful mapping back to the original (unnormalized) weekly_avg_returns
        # The `train_test_split` with `shuffle=False` ensures a contiguous block of data.
        total_sequences = len(normalized_returns) - lookback_period
        train_val_sequences_count = len(train_loader.dataset) + len(val_loader.dataset)
        # The test data starts at the index corresponding to the first sequence in the test set
        test_start_idx_in_original_df = lookback_period + train_val_sequences_count
        actual_test_period_returns_df = weekly_avg_returns.iloc[test_start_idx_in_original_df:]

        if actual_test_period_returns_df.empty:
            print("Not enough actual returns for the test period to evaluate. Adjust data range or lookback.")
            exit()

        # Evaluate NN Portfolio
        nn_portfolio_metrics = evaluate_portfolio_performance(actual_test_period_returns_df, avg_nn_weights)
        print("\nNeural Network Portfolio Performance (Average Weights on Test Period):")
        print(f"  Annualized Return: {nn_portfolio_metrics['annualized_return']:.4f}")
        print(f"  Annualized Volatility: {nn_portfolio_metrics['annualized_volatility']:.4f}")
        print(f"  Sharpe Ratio: {nn_portfolio_metrics['sharpe_ratio']:.4f}")
        print(f"  Cumulative Return: {nn_portfolio_metrics['cumulative_return']:.4f}")
        print("  Average NN Predicted Weights:", [f"{w:.4f}" for w in avg_nn_weights])


        # Evaluate MVO Target Portfolio (what the NN was trying to learn)
        mvo_portfolio_metrics = evaluate_portfolio_performance(actual_test_period_returns_df, avg_mvo_weights)
        print("\nMVO Target Portfolio Performance (Average Weights on Test Period):")
        print(f"  Annualized Return: {mvo_portfolio_metrics['annualized_return']:.4f}")
        print(f"  Annualized Volatility: {mvo_portfolio_metrics['annualized_volatility']:.4f}")
        print(f"  Sharpe Ratio: {mvo_portfolio_metrics['sharpe_ratio']:.4f}")
        print(f"  Cumulative Return: {mvo_portfolio_metrics['cumulative_return']:.4f}")
        print("  Average MVO Target Weights:", [f"{w:.4f}" for w in avg_mvo_weights])

        # --- Visualize Cumulative Returns of Portfolios ---
        nn_portfolio_returns_series = actual_test_period_returns_df.dot(avg_nn_weights)
        mvo_portfolio_returns_series = actual_test_period_returns_df.dot(avg_mvo_weights)

        cumulative_nn_portfolio_returns = (1 + nn_portfolio_returns_series).cumprod() - 1
        cumulative_mvo_portfolio_returns = (1 + mvo_portfolio_returns_series).cumprod() - 1
        # Ensure cumulative_asset_returns is aligned with the test period
        cumulative_asset_returns_test_period = (1 + actual_test_period_returns_df).cumprod() - 1


        plt.figure(figsize=(14, 8))
        plt.plot(cumulative_nn_portfolio_returns.index, cumulative_nn_portfolio_returns, label='NN Portfolio', linewidth=2.5, color='blue')
        plt.plot(cumulative_mvo_portfolio_returns.index, cumulative_mvo_portfolio_returns, label='MVO Target Portfolio', linewidth=2.5, color='green', linestyle='--')

        for ticker in actual_test_period_returns_df.columns:
            plt.plot(cumulative_asset_returns_test_period.index, cumulative_asset_returns_test_period[ticker], label=ticker, linestyle=':', alpha=0.7)

        plt.title("Cumulative Returns: NN Portfolio vs MVO Target vs Individual Assets (Test Period)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    except ValueError as e:
        print(f"Error during execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
