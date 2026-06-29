import pandas as pd
import numpy as np

predictions = pd.read_csv('data/predictions.csv')
regime = pd.read_csv('data/regime.csv', index_col=0, parse_dates=True)

predictions['date'] = pd.to_datetime(predictions['date'])
regime.index = pd.to_datetime(regime.index)

portfolio_weights = []

for date in predictions['date'].unique():
    date_pred = predictions[predictions['date'] == date]

    if date not in regime.index:
        continue

    regime_val = regime.loc[date, 'regime']
    stocks = date_pred['stock'].values
    n_stocks = len(stocks)

    w_equal = np.ones(n_stocks) / n_stocks

    signal = (date_pred['predicted_return'].values / (date_pred['uncertainty'].values + 1e-8))
    signal = np.maximum(signal, 0)
    signal = np.nan_to_num(signal, nan=0.0)

    signal_sum = signal.sum()
    if signal_sum > 1e-8:
        w_risk = signal / signal_sum
    else:
        w_risk = w_equal.copy()

    w_risk = np.nan_to_num(w_risk, nan=1/n_stocks)
    w_risk = w_risk / w_risk.sum()

    if regime_val == 'Bear':
        w_regime = w_equal.copy()
        high_unc = date_pred['uncertainty'].values > date_pred['uncertainty'].median()
        w_regime[high_unc] *= 0.5
        w_regime = w_regime / w_regime.sum()
    elif regime_val == 'Bull':
        signal_sum = signal.sum()
        if signal_sum > 1e-8:
            w_regime = signal / signal_sum
        else:
            w_regime = w_equal.copy()

        w_regime = np.nan_to_num(w_regime, nan=1/n_stocks)
        w_regime = w_regime / w_regime.sum()
    else:
        w_regime = w_equal.copy()

    for i, stock in enumerate(stocks):
        portfolio_weights.append({
            'date': date,
            'stock': stock,
            'weight_equal': w_equal[i],
            'weight_risk_adjusted': w_risk[i],
            'weight_regime_aware': w_regime[i]
        })

weights_df = pd.DataFrame(portfolio_weights)
weights_df.to_csv('data/portfolio_weights.csv', index=False)

print(f"Portfolio weights shape: {weights_df.shape}")
print(f"Date range: {weights_df['date'].min()} to {weights_df['date'].max()}")
print(f"Unique dates: {weights_df['date'].nunique()}")
print(f"Weight sums per date (equal): {weights_df.groupby('date')['weight_equal'].sum().mean():.4f}")
print(f"Weight sums per date (risk-adjusted): {weights_df.groupby('date')['weight_risk_adjusted'].sum().mean():.4f}")
print(f"Weight sums per date (regime-aware): {weights_df.groupby('date')['weight_regime_aware'].sum().mean():.4f}")
