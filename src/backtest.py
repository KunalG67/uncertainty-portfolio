import pandas as pd
import numpy as np
from pathlib import Path

def calculate_metrics(portfolio_values, rf_rate=0.02):
    """Calculate Sharpe ratio, max drawdown, annualized return, Calmar ratio."""
    returns = portfolio_values.pct_change().dropna()

    if len(returns) == 0:
        return {'Annualized Return': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Calmar Ratio': 0, 'Total Return': 0, 'Active Period (years)': 0}

    # Find first active date (when portfolio value changes from initial value)
    initial_value = portfolio_values.iloc[0]
    first_active_idx = (portfolio_values != initial_value).argmax()
    if first_active_idx == 0 and portfolio_values.iloc[0] == initial_value:
        first_active_date = portfolio_values.index[0]
    else:
        first_active_date = portfolio_values.index[first_active_idx]

    last_date = portfolio_values.index[-1]

    # Calculate n_years using only active period
    n_years = (last_date - first_active_date).days / 365.25

    # Calculate total and annualized return using correct time period
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    if n_years > 0:
        annual_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annual_return = 0

    annual_std = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - rf_rate) / annual_std if annual_std > 0 else 0

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    return {
        'Annualized Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Total Return': total_return,
        'Active Period (years)': n_years
    }

def backtest_portfolio(prices, weights_df, weight_col, initial_capital=100000, start_date=None):
    """Run backtest for a portfolio using specified weights."""
    portfolio_values = [initial_capital]
    current_value = initial_capital

    dates = prices.index
    last_weights = None
    last_stocks = None
    # Weights computed at close of day t can only be traded at open/close of day t+1.
    # pending_* holds newly-computed weights until the next iteration activates them.
    pending_weights = None
    pending_stocks = None

    TRANSACTION_COST = 0.0010  # 10 bps one-way on turnover

    sample_weights = []
    start_idx = 0

    if start_date is not None:
        start_idx = (dates >= start_date).argmax()
        portfolio_values = [initial_capital] * start_idx
        current_value = initial_capital

    for i in range(max(1, start_idx), len(dates)):
        current_date = dates[i]

        # --- Step 1: activate weights queued from the previous day's rebalance ---
        if pending_weights is not None:
            if last_weights is not None:
                old_w = dict(zip(last_stocks, last_weights))
                new_w = dict(zip(pending_stocks, pending_weights))
                all_s = set(old_w) | set(new_w)
                turnover = sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in all_s)
                current_value *= (1.0 - turnover * TRANSACTION_COST)
            last_weights = pending_weights
            last_stocks = pending_stocks
            if len(sample_weights) < 3:
                sample_weights.append({
                    'date': current_date,
                    'weight_col': weight_col,
                    'weights': last_weights.copy(),
                    'stocks': last_stocks.copy()
                })
            pending_weights = None
            pending_stocks = None

        # --- Step 2: queue new weights if today is a rebalance date ---
        # (They use features known only at today's close, so they take effect tomorrow.)
        date_weights = weights_df[weights_df['date'] == current_date.date()]
        if not date_weights.empty:
            w = date_weights[weight_col].values
            w = w / w.sum()
            pending_weights = w
            pending_stocks = date_weights['stock'].values

        # --- Step 3: compute today's return using weights decided before today ---
        if last_weights is None:
            daily_return = (prices.iloc[i] / prices.iloc[i-1]).mean() - 1
            current_value = current_value * (1 + daily_return)
            portfolio_values.append(current_value)
            continue

        available_prices = prices.columns.intersection(last_stocks)
        price_returns = (prices.iloc[i][available_prices] / prices.iloc[i-1][available_prices]).values

        stock_indices = [np.where(last_stocks == stock)[0][0] for stock in available_prices]
        weights_aligned = last_weights[stock_indices]
        weights_aligned = weights_aligned / weights_aligned.sum() if weights_aligned.sum() > 0 else weights_aligned

        portfolio_return = np.dot(price_returns - 1, weights_aligned)
        current_value = current_value * (1 + portfolio_return)
        portfolio_values.append(current_value)

    return pd.Series(portfolio_values, index=dates), sample_weights

def main():
    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    prices = pd.read_csv(data_dir / 'prices.csv', index_col=0, parse_dates=True)
    weights = pd.read_csv(data_dir / 'portfolio_weights.csv')
    weights['date'] = pd.to_datetime(weights['date']).dt.date

    prices = prices.sort_index()

    print("\n" + "="*90)
    print("DEBUG: PORTFOLIO WEIGHTS DATA")
    print("="*90)
    print(f"Weights shape: {weights.shape}")
    print(f"Weights columns: {weights.columns.tolist()}")
    print(f"Date range: {weights['date'].min()} to {weights['date'].max()}")
    print(f"Unique dates: {weights['date'].nunique()}")
    print(f"\nUnique dates per weight column:")
    for col in ['weight_equal', 'weight_risk_adjusted', 'weight_regime_aware']:
        unique_dates = weights[weights[col].notna()]['date'].nunique()
        print(f"  {col}: {unique_dates} dates")
    print(f"\nFirst 5 rows of weights:")
    print(weights.head())
    print(f"\nLast 5 rows of weights:")
    print(weights.tail())
    print("="*90 + "\n")

    valid_dates = weights['date'].unique()
    valid_dates = sorted(valid_dates)
    first_valid_date = valid_dates[0] if valid_dates else None

    print(f"\nWeight data coverage:")
    print(f"  First valid date for all strategies: {first_valid_date}")
    print(f"  Prices data from: {prices.index.min()} to {prices.index.max()}")
    print(f"  Backtest will run from: {first_valid_date}\n")

    pf_equal, samples_equal = backtest_portfolio(prices, weights, 'weight_equal', start_date=pd.to_datetime(first_valid_date))
    pf_risk, samples_risk = backtest_portfolio(prices, weights, 'weight_risk_adjusted', start_date=pd.to_datetime(first_valid_date))
    pf_regime, samples_regime = backtest_portfolio(prices, weights, 'weight_regime_aware', start_date=pd.to_datetime(first_valid_date))

    pf_equal_returns = pf_equal.pct_change().dropna()
    pf_risk_returns = pf_risk.pct_change().dropna()
    pf_regime_returns = pf_regime.pct_change().dropna()

    print("\n" + "="*90)
    print("DEBUG: DAILY PORTFOLIO RETURNS")
    print("="*90)
    print(f"\nEqual Weight - First 5 daily returns:")
    print(pf_equal_returns.head())
    print(f"\nRisk-Adjusted - First 5 daily returns:")
    print(pf_risk_returns.head())
    print(f"\nRegime-Aware - First 5 daily returns:")
    print(pf_regime_returns.head())
    print("="*90 + "\n")

    print("\n" + "="*90)
    print("DEBUG: SAMPLE WEIGHTS USED IN EACH PORTFOLIO")
    print("="*90)
    print("\nEqual Weight samples:")
    for s in samples_equal:
        print(f"  Date {s['date']}: weights sum={s['weights'].sum():.6f}, first 3 weights={s['weights'][:3]}")
    print("\nRisk-Adjusted samples:")
    for s in samples_risk:
        print(f"  Date {s['date']}: weights sum={s['weights'].sum():.6f}, first 3 weights={s['weights'][:3]}")
    print("\nRegime-Aware samples:")
    for s in samples_regime:
        print(f"  Date {s['date']}: weights sum={s['weights'].sum():.6f}, first 3 weights={s['weights'][:3]}")
    print("="*90 + "\n")

    print("\n" + "="*90)
    print("DEBUG: PORTFOLIO VALUES")
    print("="*90)
    print(f"\nEqual Weight - First 10 values:")
    print(pf_equal.head(10))
    print(f"\nEqual Weight - Last 10 values:")
    print(pf_equal.tail(10))
    print(f"\nRisk-Adjusted - First 10 values:")
    print(pf_risk.head(10))
    print(f"\nRisk-Adjusted - Last 10 values:")
    print(pf_risk.tail(10))
    print(f"\nRegime-Aware - First 10 values:")
    print(pf_regime.head(10))
    print(f"\nRegime-Aware - Last 10 values:")
    print(pf_regime.tail(10))
    print("="*90 + "\n")

    results_df = pd.DataFrame({
        'Date': prices.index,
        'Equal Weight': pf_equal.values,
        'Risk-Adjusted': pf_risk.values,
        'Regime-Aware': pf_regime.values
    })
    results_df.to_csv(results_dir / 'backtest_results.csv', index=False)

    metrics = []
    for name, pf_values in [
        ('Equal Weight', pf_equal),
        ('Risk-Adjusted', pf_risk),
        ('Regime-Aware', pf_regime)
    ]:
        m = calculate_metrics(pf_values)
        m['Portfolio'] = name
        metrics.append(m)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[['Portfolio', 'Total Return', 'Active Period (years)', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']]

    print("\n" + "="*90)
    print("BACKTEST RESULTS - PORTFOLIO PERFORMANCE")
    print("="*90)

    for idx, row in metrics_df.iterrows():
        print(f"\n{row['Portfolio']}")
        print(f"  Total Return:       {row['Total Return']:>10.2%}")
        print(f"  Active Period:      {row['Active Period (years)']:>10.2f} years")
        print(f"  Annualized Return:  {row['Annualized Return']:>10.2%}")
        print(f"  Sharpe Ratio:       {row['Sharpe Ratio']:>10.2f}")
        print(f"  Max Drawdown:       {row['Max Drawdown']:>10.2%}")
        print(f"  Calmar Ratio:       {row['Calmar Ratio']:>10.2f}")

    print("\n" + "="*90)
    print(f"Results saved to {results_dir / 'backtest_results.csv'}")
    print("="*90 + "\n")

if __name__ == '__main__':
    main()
