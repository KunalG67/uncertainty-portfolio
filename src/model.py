import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
regime = pd.read_csv('data/regime.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)

data = features.join([regime, returns], how='inner')
data = data.dropna()

stock_cols = [col for col in returns.columns if col not in ['^VIX', 'GLD', 'USO']]

regime_dummies = pd.get_dummies(data['regime'], prefix='regime')
X = data.drop(columns=['regime'] + stock_cols).join(regime_dummies)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

predictions = []

for stock in stock_cols:
    y = returns[stock].loc[X.index]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    tree_preds = np.array([tree.predict(X_test_scaled) for tree in model.estimators_])
    uncertainty = np.std(tree_preds, axis=0)

    for i, date in enumerate(X_test.index):
        predictions.append({
            'date': date,
            'stock': stock,
            'predicted_return': preds[i],
            'uncertainty': uncertainty[i]
        })

pred_df = pd.DataFrame(predictions)
pred_df.to_csv('data/predictions.csv', index=False)

print(f"Predictions shape: {pred_df.shape}")
print(f"Mean uncertainty: {pred_df['uncertainty'].mean():.4f}")
