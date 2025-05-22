import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import jarque_bera, normaltest
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

torch.manual_seed(42)
np.random.seed(42)

# --- Advanced Statistical Features ---
class AdvancedFeatureExtractor:
    """Extract advanced statistical and numerical features for portfolio optimization"""
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20]  # Different time horizons
        
    def extract_statistical_features(self, returns_df):
        """Extract advanced statistical features"""
        features = []
        
        # Basic moments for each asset
        features.extend(returns_df.mean().values)  # Mean returns
        features.extend(returns_df.std().values)   # Volatility
        features.extend(returns_df.skew().values)  # Skewness
        features.extend(returns_df.kurtosis().values)  # Kurtosis
        
        # Tail risk measures
        features.extend(returns_df.quantile(0.05).values)  # VaR 95%
        
        # Rolling statistics for different windows
        for window in self.lookback_periods:
            if len(returns_df) >= window:
                rolling_vol = returns_df.rolling(window=window).std().iloc[-1]
                features.extend(rolling_vol.fillna(0).values)
                
                rolling_mean = returns_df.rolling(window=window).mean().iloc[-1]
                features.extend(rolling_mean.fillna(0).values)
        
        # Correlation features (upper triangular)
        corr_matrix = returns_df.corr()
        n_assets = len(returns_df.columns)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                features.append(corr_matrix.iloc[i, j])
        
        # Handle any NaN values
        features = [0.0 if pd.isna(x) else float(x) for x in features]
        
        return np.array(features)

# --- Robust Covariance Estimation ---
class RobustCovarianceEstimator:
    """Advanced covariance estimation methods"""
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns_df):
        """Ledoit-Wolf shrinkage estimator"""
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns_df)
            return lw.covariance_, lw.shrinkage_
        except:
            # Fallback to regular covariance
            return returns_df.cov().values, 0.0
    
    @staticmethod
    def exponential_weighted_cov(returns_df, alpha=0.94):
        """Exponentially weighted covariance matrix"""
        return returns_df.ewm(alpha=alpha).cov().iloc[-len(returns_df.columns):, :]

# --- Enhanced Neural Network Architecture ---
class EnhancedPortfolioNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64, 32]):
        super(EnhancedPortfolioNN, self).__init__()
        
        print(f"Initializing NN with input_size={input_size}, output_size={output_size}")
        
        # Feature extraction layers
        self.feature_layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Main processing layers
        layers = []
        current_size = hidden_sizes[0]
        
        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_size = hidden_size
        
        self.main_network = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(current_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure input is properly shaped
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        
        # Feature extraction
        features = self.feature_layer1(x)
        
        # Main processing
        processed = self.main_network(features)
        
        # Generate weights
        raw_weights = self.output_layer(processed)
        weights = self.softmax(raw_weights)
        
        return weights

# --- Advanced Loss Functions ---
class AdvancedLossFunctions:
    """Collection of advanced loss functions for portfolio optimization"""
    
    @staticmethod
    def sharpe_ratio_loss(predicted_weights, returns_tensor, risk_free_rate=0.0):
        """Loss based on Sharpe ratio maximization"""
        # Calculate portfolio returns
        portfolio_returns = torch.sum(predicted_weights * returns_tensor, dim=1)
        
        # Calculate mean and std
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # Sharpe ratio (negative for minimization)
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        return -sharpe_ratio
    
    @staticmethod
    def portfolio_variance_loss(predicted_weights, cov_matrix):
        """Portfolio variance loss"""
        batch_size = predicted_weights.size(0)
        total_variance = 0
        
        for i in range(batch_size):
            w = predicted_weights[i].unsqueeze(0)  # Shape: (1, n_assets)
            variance = torch.mm(torch.mm(w, cov_matrix), w.t())
            total_variance += variance
            
        return total_variance / batch_size

# --- Data Preparation with Advanced Features ---
def prepare_enhanced_nn_data(normalized_returns_df, raw_returns_df, lookback_period=52, 
                           validation_split=0.15, test_split=0.2):
    """Enhanced data preparation with advanced statistical features"""
    print(f"\nPreparing enhanced NN data with lookback_period={lookback_period}...")
    
    feature_extractor = AdvancedFeatureExtractor()
    
    sequences = []
    targets = []
    
    # Create sequences
    for i in range(lookback_period, len(raw_returns_df)):
        # Historical window
        history_window = raw_returns_df.iloc[i - lookback_period:i]
        
        # Basic sequence (flattened returns)
        basic_sequence = normalized_returns_df.iloc[i - lookback_period:i].values.flatten()
        
        # Advanced features
        advanced_features = feature_extractor.extract_statistical_features(history_window)
        
        # Combine features
        full_sequence = np.concatenate([basic_sequence, advanced_features])
        sequences.append(full_sequence)
        
        # Target: next period returns
        if i < len(raw_returns_df):
            next_returns = raw_returns_df.iloc[i].values
            targets.append(next_returns)
    
    # Convert to numpy arrays
    X = np.array(sequences[:-1])  # Remove last sequence as it has no target
    y = np.array(targets)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Raw feature dimension: {X.shape[1]}")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    n_samples = len(X_scaled)
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * validation_split)
    train_size = n_samples - test_size - val_size
    
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size + val_size]
    X_test = X_scaled[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

# --- Enhanced Mean-Variance Optimization ---
def enhanced_mean_variance_optimization(returns_df, risk_aversion=2.0, use_robust_cov=True):
    """Enhanced mean-variance optimization with robust covariance estimation"""
    print("\nRunning Enhanced Mean-Variance Optimization...")
    
    # Calculate expected returns (using exponential weighting)
    expected_returns = returns_df.ewm(span=63).mean().iloc[-1].values  # ~3 months
    
    # Robust covariance estimation
    if use_robust_cov:
        cov_estimator = RobustCovarianceEstimator()
        cov_matrix, shrinkage = cov_estimator.ledoit_wolf_shrinkage(returns_df)
        print(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")
    else:
        cov_matrix = returns_df.cov().values
    
    n_assets = len(expected_returns)
    
    # Optimization variables
    w = cp.Variable(n_assets)
    
    # Objective: maximize utility (expected return - risk penalty)
    portfolio_return = expected_returns.T @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,     # Fully invested
        w >= 0.01,          # Minimum 1% in each asset
        w <= 0.5            # Maximum 50% in any single asset
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights = w.value
        
        # Calculate portfolio metrics
        portfolio_return_val = np.dot(expected_returns, optimal_weights)
        portfolio_risk_val = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return_val / portfolio_risk_val if portfolio_risk_val > 0 else 0
        
        print(f"Enhanced MVO - Expected Return: {portfolio_return_val:.4f}")
        print(f"Enhanced MVO - Risk (Std): {portfolio_risk_val:.4f}")
        print(f"Enhanced MVO - Sharpe Ratio: {sharpe_ratio:.4f}")
        
        return optimal_weights, cov_matrix
    else:
        print(f"Optimization failed with status: {problem.status}")
        # Return equal weights as fallback
        return np.ones(n_assets) / n_assets, np.eye(n_assets)

# --- Training Function ---
def train_enhanced_portfolio_nn(model, train_data, val_data, epochs=100, learning_rate=0.001, 
                              patience=15, returns_df=None):
    """Enhanced training with multiple loss functions and early stopping"""
    print("\nTraining Enhanced Portfolio Neural Network...")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True)
    
    # Loss functions
    loss_functions = AdvancedLossFunctions()
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Calculate covariance matrix for loss function
    if returns_df is not None:
        cov_matrix = torch.FloatTensor(returns_df.cov().values).to(device)
    else:
        cov_matrix = torch.eye(y_train.shape[1]).to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_epoch = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predicted_weights = model(batch_X)
            
            # Combined loss function
            # 1. Ensure weights sum to 1 (already handled by softmax)
            # 2. Minimize portfolio variance
            variance_loss = loss_functions.portfolio_variance_loss(predicted_weights, cov_matrix)
            
            # 3. Sharpe ratio loss (if we have enough samples)
            if len(batch_y) > 5:
                sharpe_loss = loss_functions.sharpe_ratio_loss(predicted_weights, batch_y)
                total_loss = 0.6 * variance_loss + 0.4 * sharpe_loss
            else:
                total_loss = variance_loss
            
            # Add regularization to prevent extreme weights
            weight_penalty = torch.mean(torch.sum(predicted_weights ** 2, dim=1))
            total_loss += 0.01 * weight_penalty
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predicted_weights = model(batch_X)
                
                variance_loss = loss_functions.portfolio_variance_loss(predicted_weights, cov_matrix)
                
                if len(batch_y) > 5:
                    sharpe_loss = loss_functions.sharpe_ratio_loss(predicted_weights, batch_y)
                    total_loss = 0.6 * variance_loss + 0.4 * sharpe_loss
                else:
                    total_loss = variance_loss
                
                weight_penalty = torch.mean(torch.sum(predicted_weights ** 2, dim=1))
                total_loss += 0.01 * weight_penalty
                
                val_loss_epoch += total_loss.item()
        
        # Average losses
        train_loss_avg = train_loss_epoch / len(train_loader)
        val_loss_avg = val_loss_epoch / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_avg:.6f}, "
                  f"Val Loss: {val_loss_avg:.6f}, LR: {current_lr:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# --- Portfolio Evaluation ---
def evaluate_portfolio_performance(returns_df, weights_dict, start_date=None, end_date=None):
    """Comprehensive portfolio performance evaluation"""
    print("\nEvaluating Portfolio Performance...")
    
    if start_date or end_date:
        returns_df = returns_df.loc[start_date:end_date]
    
    results = {}
    
    for method_name, weights in weights_dict.items():
        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(weights)
        
        # Performance metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Risk metrics
        var_95 = portfolio_returns.quantile(0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        results[method_name] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Max Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Weights': weights
        }
    
    return results

# --- Main Execution ---
def main():
    # Configuration
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    print("=== Enhanced Portfolio Optimization System ===")
    print(f"Assets: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    
    # Download data
    print("\nDownloading market data...")
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        price_data = price_data.dropna()
        
        # Calculate returns
        returns_data = price_data.pct_change().dropna()
        
        print(f"Data shape: {price_data.shape}")
        print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Normalize returns for NN training
    scaler = RobustScaler()
    normalized_returns = pd.DataFrame(
        scaler.fit_transform(returns_data),
        index=returns_data.index,
        columns=returns_data.columns
    )
    
    # Prepare data for neural network
    train_data, val_data, test_data, feature_scaler = prepare_enhanced_nn_data(
        normalized_returns, returns_data, 
        lookback_period=40,  # Reduced for more training samples
        validation_split=0.15,
        test_split=0.2
    )
    
    # Initialize neural network
    input_size = train_data[0].shape[1]
    output_size = len(tickers)
    
    model = EnhancedPortfolioNN(
        input_size=input_size, 
        output_size=output_size,
        hidden_sizes=[128, 64, 32]  # Smaller network
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train model
    train_losses, val_losses = train_enhanced_portfolio_nn(
        model, train_data, val_data, 
        epochs=100, learning_rate=0.001,
        returns_df=returns_data
    )
    
    # Get neural network predictions
    model.eval()
    with torch.no_grad():
        # Use last validation sample
        sample_input = torch.FloatTensor(val_data[0][-1:]).to(device)
        nn_weights = model(sample_input).cpu().numpy()[0]
    
    # Enhanced Mean-Variance Optimization
    mvo_weights, cov_matrix = enhanced_mean_variance_optimization(
        returns_data.iloc[-252:],  # Use last year of data
        risk_aversion=2.0,
        use_robust_cov=True
    )
    
    # Equal weights benchmark
    equal_weights = np.ones(len(tickers)) / len(tickers)
    
    # Compile results
    weights_dict = {
        'Neural Network': nn_weights,
        'Enhanced MVO': mvo_weights,
        'Equal Weight': equal_weights
    }
    
    # Evaluate performance on test data
    test_returns = returns_data.iloc[-len(test_data[0]):]
    performance_results = evaluate_portfolio_performance(test_returns, weights_dict)
    
    # Display results
    print("\n" + "="*80)
    print("PORTFOLIO PERFORMANCE COMPARISON (Test Period)")
    print("="*80)
    
    results_df = pd.DataFrame(performance_results).T
    print(results_df.round(4))
    
    # Print individual weights
    print("\n" + "="*80)
    print("PORTFOLIO WEIGHTS")
    print("="*80)
    
    weights_df = pd.DataFrame({
        'Neural Network': nn_weights,
        'Enhanced MVO': mvo_weights,
        'Equal Weight': equal_weights
    }, index=tickers)
    
    print(weights_df.round(4))
    
    print(f"\nTraining completed in {len(train_losses)} epochs")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    print("\n=== Analysis Complete ===")
    
    return {
        'price_data': price_data,
        'returns_data': returns_data,
        'weights_dict': weights_dict,
        'performance_results': performance_results,
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

if __name__ == "__main__":
    results = main()