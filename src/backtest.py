import pandas as pd
import numpy as np
from pathlib import Path

def calculate_metrics(portfolio_values, rf_rate=0.02):
    """Calculate Sharpe ratio, max drawdown, annualized return, Calmar ratio."""
    returns = portfolio_values.pct_change().dropna()

    if len(returns) == 0:
        return {'Annualized Return': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Calmar Ratio': 0}

    annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(returns)) - 1
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
        'Calmar Ratio': calmar_ratio
    }

def backtest_portfolio(prices, weights_df, weight_col, initial_capital=100000):
    """Run backtest for a portfolio using specified weights."""
    portfolio_values = [initial_capital]
    current_value = initial_capital

    dates = prices.index
    last_rebalance_month = None

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        if current_date.date() not in weights_df['date'].values:
            daily_return = (prices.iloc[i] / prices.iloc[i-1]).mean() - 1
            current_value = current_value * (1 + daily_return)
            portfolio_values.append(current_value)
            continue

        date_weights = weights_df[weights_df['date'] == current_date.date()]

        if date_weights.empty:
            daily_return = (prices.iloc[i] / prices.iloc[i-1]).mean() - 1
            current_value = current_value * (1 + daily_return)
            portfolio_values.append(current_value)
            continue

        stocks = date_weights['stock'].values
        weights = date_weights[weight_col].values
        weights = weights / weights.sum()

        available_prices = prices.columns.intersection(stocks)
        price_returns = (prices.iloc[i][available_prices] / prices.iloc[i-1][available_prices]).values
        weights_aligned = weights[:len(available_prices)]
        weights_aligned = weights_aligned / weights_aligned.sum() if weights_aligned.sum() > 0 else weights_aligned

        portfolio_return = np.dot(price_returns - 1, weights_aligned)
        current_value = current_value * (1 + portfolio_return)
        portfolio_values.append(current_value)

    return pd.Series(portfolio_values, index=dates)

def main():
    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    prices = pd.read_csv(data_dir / 'prices.csv', index_col=0, parse_dates=True)
    weights = pd.read_csv(data_dir / 'portfolio_weights.csv')
    weights['date'] = pd.to_datetime(weights['date']).dt.date

    prices = prices.sort_index()

    pf_equal = backtest_portfolio(prices, weights, 'weight_equal')
    pf_risk = backtest_portfolio(prices, weights, 'weight_risk_adjusted')
    pf_regime = backtest_portfolio(prices, weights, 'weight_regime_aware')

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
    metrics_df = metrics_df[['Portfolio', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']]

    print("\n" + "="*90)
    print("BACKTEST RESULTS - PORTFOLIO PERFORMANCE")
    print("="*90)

    for idx, row in metrics_df.iterrows():
        print(f"\n{row['Portfolio']}")
        print(f"  Annualized Return:  {row['Annualized Return']:>10.2%}")
        print(f"  Sharpe Ratio:       {row['Sharpe Ratio']:>10.2f}")
        print(f"  Max Drawdown:       {row['Max Drawdown']:>10.2%}")
        print(f"  Calmar Ratio:       {row['Calmar Ratio']:>10.2f}")

    print("\n" + "="*90)
    print(f"Results saved to {results_dir / 'backtest_results.csv'}")
    print("="*90 + "\n")

if __name__ == '__main__':
    main()
