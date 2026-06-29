import yfinance as yf
import pandas as pd

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC',
    'JNJ', 'PFE', 'XOM', 'CVX', 'WMT',
    'HD', 'PG', 'KO', 'DIS', 'TSLA',
    '^VIX', 'GLD', 'USO'
]

start_date = '2018-01-01'
end_date = '2024-01-01'

data = yf.download(tickers, start=start_date, end=end_date)['Close']
data.columns = [col if isinstance(col, str) else col[0] for col in data.columns]

data.to_csv('data/prices.csv')

returns = data.pct_change().dropna()
returns.to_csv('data/returns.csv')

print(f"Downloaded data for {len(tickers)} assets")
print(f"Price data shape: {data.shape}")
print(f"Returns data shape: {returns.shape}")
