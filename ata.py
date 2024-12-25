import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Step 1: Load the Dataset
def load_data(file_path):
    """Load cryptocurrency data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure the file exists.")
    data = pd.read_csv(file_path)
    if 'Close' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
        raise ValueError("Dataset must contain 'Close', 'High', and 'Low' columns.")
    return data

# Step 2: Preprocess the Data
def preprocess_data(data):
    """Preprocess data: handle missing values and normalize prices."""
    data = data.dropna()
    scaler = MinMaxScaler()
    data['Normalized_Price'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data

# Step 3: Identify Arbitrage Opportunities
def find_arbitrage_opportunities(data, threshold=0.02):
    """Find arbitrage opportunities based on price differences across exchanges."""
    data['Price_Diff'] = data['High'] - data['Low']
    data['Arbitrage'] = data['Price_Diff'] > (data['Close'] * threshold)
    return data

# Step 4: Backtest the Trading Strategy
def backtest_strategy(data, fee_rate=0.001):
    """Simulate trades based on identified arbitrage opportunities."""
    initial_balance = 1000  # Starting with $1,000
    balance = initial_balance
    trades = 0

    for index, row in data.iterrows():
        if row['Arbitrage']:
            # Simulate a trade: buy low, sell high, accounting for fees
            trade_profit = (row['Price_Diff'] - (row['Close'] * fee_rate * 2))  # Buy and sell fees
            if trade_profit > 0:
                balance += trade_profit
                trades += 1

    return balance, trades

# Step 5: Visualize the Results
def visualize_results(data):
    """Plot price and arbitrage opportunities."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Normalized_Price'], label='Normalized Price', linewidth=1)
    plt.scatter(data.index[data['Arbitrage']], data['Normalized_Price'][data['Arbitrage']],
                color='red', label='Arbitrage Opportunities', alpha=0.5)
    plt.title('Price Trends and Arbitrage Opportunities')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid()
    plt.show()

# Step 6: Save Results
def save_results(final_balance, total_trades, output_file="results.txt"):
    """Save the backtesting results to a file."""
    with open(output_file, 'w') as file:
        file.write(f"Final Balance: ${final_balance:.2f}\n")
        file.write(f"Total Trades Executed: {total_trades}\n")
    print(f"Results saved to {output_file}")

# Main Function
def main():
    file_path = 'data/coin_Bitcoin.csv'
    try:
        data = load_data(file_path)
        data = preprocess_data(data)
        data = find_arbitrage_opportunities(data)
        final_balance, total_trades = backtest_strategy(data)

        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Trades Executed: {total_trades}")

        visualize_results(data)
        save_results(final_balance, total_trades)
    except (FileNotFoundError, ValueError) as e:
        print(e)

if __name__ == "__main__":
    main()
