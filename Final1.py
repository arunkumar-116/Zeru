import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
TOKEN_DECIMALS = {
    'USDC': 6, 'USDT': 6, 'DAI': 18,
    'WETH': 18, 'WMATIC': 18, 'WBTC': 8
}

def process_transactions(input_file, output_file='credit_scores.csv'):
    """End-to-end credit scoring pipeline using RandomForest"""
    
    # 1. Load and preprocess data
    print(f" Loading data from {input_file}...")
    with open(input_file) as f:
        transactions = json.load(f)
    
    df = pd.json_normalize(transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 2. Normalize token amounts
    def normalize_amount(row):
        asset = row.get('actionData.assetSymbol', '')
        decimals = TOKEN_DECIMALS.get(asset, 18)
        amount = float(row.get('actionData.amount', 0))
        return amount / (10 ** decimals)
    
    df['normalized_amount'] = df.apply(normalize_amount, axis=1)
    df['usd_value'] = df['normalized_amount'] * df['actionData.assetPriceUSD'].astype(float)
    
    # 3. Extract wallet features
    print("Calculating wallet features...")
    wallet_features = []
    
    for wallet, group in df.groupby('userWallet'):
        time_diff = (group['timestamp'].max() - group['timestamp'].min()).days + 1
        
        wallet_features.append({
            'wallet': wallet,
            'tx_count': len(group),
            'deposit_usd': group[group['action'] == 'deposit']['usd_value'].sum(),
            'redeem_usd': group[group['action'] == 'redeemunderlying']['usd_value'].sum(),
            'borrow_usd': group[group['action'] == 'borrow']['usd_value'].sum() if 'borrow' in group['action'].values else 0,
            'repay_usd': group[group['action'] == 'repay']['usd_value'].sum(),
            'liquidation_count': (group['action'] == 'liquidationcall').sum(),
            'active_days': time_diff,
            'unique_tokens': group['actionData.assetSymbol'].nunique(),
            'tx_freq': len(group) / time_diff
        })
    
    wallet_df = pd.DataFrame(wallet_features).fillna(0)

    # 4. Create synthetic credit score (target) for training
    def generate_score(row):
        score = (
            + row['repay_usd'] * 0.4
            - row['borrow_usd'] * 0.3
            + row['deposit_usd'] * 0.2
            - row['liquidation_count'] * 100
            + 100
        )
        return min(1000, max(0, round(score)))
    
    wallet_df['synthetic_score'] = wallet_df.apply(generate_score, axis=1)

    # 5. Train Random Forest model
    print("Training Random Forest model...")
    features = wallet_df.drop(['wallet', 'synthetic_score'], axis=1)
    target = wallet_df['synthetic_score']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n Evaluation Metrics:\nRMSE: {rmse:.2f} | R² Score: {r2:.2f}")

    # 6. Predict scores for all wallets
    wallet_df['credit_score'] = model.predict(X_scaled).round().clip(0, 1000)
    
    # 7. Save results
    wallet_df[['wallet', 'credit_score']].to_csv(output_file, index=False)
    print(f"\n Saved credit scores to {output_file}")
    print(f"   Wallets processed: {len(wallet_df)}")
    print(f"   Score range: {wallet_df['credit_score'].min()}–{wallet_df['credit_score'].max()}")
    
    return wallet_df

# Run the pipeline
if __name__ == '__main__':
    import sys
    input_json = sys.argv[1] if len(sys.argv) > 1 else 'user-wallet-transactions.json'
    process_transactions(input_json)