import pandas as pd
import numpy as np

prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)

stock_cols = [col for col in returns.columns if col not in ['^VIX', 'GLD', 'USO']]
avg_returns = returns[stock_cols].mean(axis=1)

rolling_return = avg_returns.rolling(50).sum()
vix = prices['^VIX']

regime = pd.Series(index=prices.index, dtype=str)

for date in regime.index:
    if date not in rolling_return.index or date not in vix.index:
        regime[date] = np.nan
        continue

    ret = rolling_return[date]
    vix_val = vix[date]

    if pd.isna(ret) or pd.isna(vix_val):
        regime[date] = np.nan
    elif ret > 0.01 and vix_val < 20:
        regime[date] = 'Bull'
    elif ret < -0.01 or vix_val > 30:
        regime[date] = 'Bear'
    else:
        regime[date] = 'Sideways'

regime_df = pd.DataFrame({'regime': regime}).dropna()
regime_df.to_csv('data/regime.csv')

print(f"Regime data shape: {regime_df.shape}")
print(f"Regime distribution:\n{regime_df['regime'].value_counts()}")
