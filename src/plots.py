import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def plot_portfolio_comparison():
    """Portfolio cumulative returns with regime background shading."""
    results = pd.read_csv('results/backtest_results.csv', parse_dates=['Date'])
    regime = pd.read_csv('data/regime.csv', index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    results_sorted = results.sort_values('Date')
    results_sorted['Date'] = pd.to_datetime(results_sorted['Date'])

    regime_colors = {'Bull': '#90EE90', 'Bear': '#FF6B6B', 'Sideways': '#D3D3D3'}
    regime_sorted = regime.sort_index()

    start_date = results_sorted['Date'].min()
    end_date = results_sorted['Date'].max()
    regime_filtered = regime_sorted[
        (regime_sorted.index >= start_date) & (regime_sorted.index <= end_date)
    ]

    regime_smoothed = regime_filtered['regime'].rolling(window=5, center=True).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[2], raw=False
    )

    current_regime = None
    regime_start = start_date

    for i, date in enumerate(regime_smoothed.index):
        if pd.isna(regime_smoothed.iloc[i]):
            continue

        next_regime = regime_smoothed.iloc[i]
        if next_regime != current_regime and current_regime is not None:
            ax.axvspan(regime_start, date, alpha=0.2, color=regime_colors[current_regime])
            regime_start = date
        current_regime = next_regime

    if current_regime is not None:
        ax.axvspan(regime_start, end_date, alpha=0.2, color=regime_colors[current_regime])

    ax.plot(results_sorted['Date'], results_sorted['Equal Weight'], label='Equal Weight', linewidth=2)
    ax.plot(results_sorted['Date'], results_sorted['Risk-Adjusted'], label='Risk-Adjusted', linewidth=2)
    ax.plot(results_sorted['Date'], results_sorted['Regime-Aware'], label='Regime-Aware', linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Portfolio Comparison with Regime Background', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    bull_patch = mpatches.Patch(color='#90EE90', label='Bull', alpha=0.3)
    bear_patch = mpatches.Patch(color='#FF6B6B', label='Bear', alpha=0.3)
    sideways_patch = mpatches.Patch(color='#D3D3D3', label='Sideways', alpha=0.3)
    ax.legend(handles=[bull_patch, bear_patch, sideways_patch], loc='upper right', fontsize=10)

    fig.tight_layout()
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    fig.savefig('results/plots/portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_analysis():
    """Scatter plot of predicted return vs uncertainty."""
    predictions = pd.read_csv('data/predictions.csv')

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(predictions['uncertainty'], predictions['predicted_return'],
                        alpha=0.5, s=30, c=predictions['predicted_return'],
                        cmap='RdYlGn', edgecolors='none')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Uncertainty (Std Dev of Tree Predictions)', fontsize=12)
    ax.set_ylabel('Predicted Return', fontsize=12)
    ax.set_title('Predicted Return vs Uncertainty Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Predicted Return', fontsize=11)

    fig.tight_layout()
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    fig.savefig('results/plots/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_distribution():
    """Bar chart of regime day counts."""
    regime = pd.read_csv('data/regime.csv', index_col=0, parse_dates=True)

    regime_counts = regime['regime'].value_counts()
    regime_order = ['Bull', 'Bear', 'Sideways']
    regime_counts = regime_counts.reindex([r for r in regime_order if r in regime_counts.index])

    colors = {'Bull': '#90EE90', 'Bear': '#FF6B6B', 'Sideways': '#D3D3D3'}
    bar_colors = [colors[regime] for regime in regime_counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(regime_counts.index, regime_counts.values, color=bar_colors, alpha=0.8, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Days', fontsize=12)
    ax.set_title('Market Regime Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    fig.savefig('results/plots/regime_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating plots...")
    plot_portfolio_comparison()
    print("✓ portfolio_comparison.png")
    plot_uncertainty_analysis()
    print("✓ uncertainty_analysis.png")
    plot_regime_distribution()
    print("✓ regime_distribution.png")
    print("\nAll plots saved to results/plots/")

if __name__ == '__main__':
    main()
