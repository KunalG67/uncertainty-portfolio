import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)
prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)

features = pd.DataFrame(index=returns.index)

stock_cols = [col for col in returns.columns if col not in ['^VIX', 'GLD', 'USO']]

for stock in stock_cols:
    features[f'{stock}_ma5'] = returns[stock].rolling(5).mean()
    features[f'{stock}_ma20'] = returns[stock].rolling(20).mean()
    features[f'{stock}_ma50'] = returns[stock].rolling(50).mean()

    features[f'{stock}_mom5'] = prices[stock].pct_change(5)
    features[f'{stock}_mom21'] = prices[stock].pct_change(21)

    features[f'{stock}_vol21'] = returns[stock].rolling(21).std()

    features[f'{stock}_rsi'] = calculate_rsi(prices[stock], 14)

features['VIX'] = prices['^VIX']

features = features.dropna()
features.to_csv('data/features.csv')

print(f"Features shape: {features.shape}")
print(f"Features created for {len(stock_cols)} stocks")
