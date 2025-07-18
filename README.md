# DeFi Wallet Credit Scoring

This project provides a machine learning-based pipeline to generate credit scores for DeFi user wallets using historical transaction data from Aave V2. The goal is to build a behavioral scoring model that predicts a wallet’s creditworthiness using transaction-level features derived from on-chain data.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Requirements](#data-requirements)
- [Feature Engineering](#feature-engineering)
- [Synthetic Score Logic](#synthetic-score-logic)
- [Machine Learning Model](#machine-learning-model)
- [Pipeline Workflow](#pipeline-workflow)
- [Usage Instructions](#usage-instructions)
- [Output](#output)
- [Evaluation Metrics](#evaluation-metrics)
- [Extensibility](#extensibility)
- [License](#license)

## Project Overview

This project processes DeFi wallet transaction data, extracts behavioral features, creates a rule-based synthetic score for training, and builds a supervised learning model using `RandomForestRegressor` to assign credit scores between 0 and 1000 to each wallet.

The pipeline is schema-agnostic, meaning it does not rely on hardcoded column names or assumptions about the dataset beyond a minimal structure required for wallet-level transaction analysis.

## Data Requirements

The input file must be a JSON file with transaction records containing the following fields per transaction:

- `userWallet`: Address of the wallet performing the transaction
- `timestamp`: Unix timestamp of the transaction
- `action`: Type of transaction (e.g., deposit, borrow, repay)
- `actionData.assetSymbol`: Token symbol used in the transaction
- `actionData.amount`: Raw amount (unadjusted for token decimals)
- `actionData.assetPriceUSD`: Price of the asset in USD at the time of the transaction

## Feature Engineering

Wallet-level features are extracted by grouping all transactions per user wallet and summarizing relevant behavior metrics:

| Feature             | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `tx_count`          | Total number of transactions                                  |
| `deposit_usd`       | Total USD value deposited                                     |
| `redeem_usd`        | Total USD value withdrawn using `redeemunderlying` actions    |
| `borrow_usd`        | Total USD value borrowed                                      |
| `repay_usd`         | Total USD value repaid                                        |
| `liquidation_count` | Number of liquidation events experienced by the wallet        |
| `active_days`       | Duration between the first and last transaction (in days)     |
| `unique_tokens`     | Number of unique tokens the wallet interacted with            |
| `tx_freq`           | Average number of transactions per day during active period   |

### Token Normalization

To ensure consistent value computation across tokens with different decimal standards, all token amounts are normalized using predefined decimal mappings:

```python
TOKEN_DECIMALS = {
    'USDC': 6, 'USDT': 6, 'DAI': 18,
    'WETH': 18, 'WMATIC': 18, 'WBTC': 8
}
```
The normalized amount is computed as:
```
normalized_amount = raw_amount / (10 ** token_decimals)
```

USD value of each transaction is calculated as:
```
usd_value = normalized_amount * asset_price_usd
```

## Synthetic Score Logic
A synthetic credit score is calculated to simulate ground truth behavior for supervised model training. The score rewards healthy financial behavior (repayment, deposits) and penalizes risky behavior (borrowing, liquidations).
```
score = (
    + 0.4 * repay_usd
    - 0.3 * borrow_usd
    + 0.2 * deposit_usd
    - 100 * liquidation_count
    + 100
)
```
The final score is clipped between 0 and 1000:
```
score = min(1000, max(0, round(score)))
```

## Machine Learning Model
The pipeline uses a `RandomForestRegressor` to learn credit behavior from wallet features and predict credit scores for each wallet.

Configuration:
Model: `RandomForestRegressor(n_estimators=100)`

Scaler: `StandardScaler` applied to all features

Training-Test Split: 80/20

Evaluation Metrics: Root Mean Squared Error (RMSE), R² Score

Model Objective:
Learn from synthetic scoring rules and generalize the scoring logic to unseen wallets based on transaction history.

## Pipeline Workflow
- Load JSON file and parse transactions.

- Normalize token amounts and compute USD value per transaction.

- Group transactions by userWallet and calculate wallet-level features.

- Generate synthetic credit scores using a rule-based function.

- Train a Random Forest model using standardized features.

- Predict credit scores for all wallets in the dataset.

- Export results to a CSV file.

## Usage Instructions
Prerequisites
Install the required Python libraries:
```python
pip install pandas numpy scikit-learn
```

Run the Pipeline
```python
python Final1.py user-wallet-transactions.json
```
If no input file is provided, it will default to `user-wallet-transactions.json`.

## Output
The output is a CSV file named credit_scores.csv containing:


| Column           | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `wallet`          | Wallet address                                |
| `credit_score`       | Predicted credit score (0–1000)                                    |

## Evaluation Metrics
Based on sample runs:

RMSE (Root Mean Squared Error): 85.34

R² Score: 0.95

These metrics indicate that the model accurately approximates the synthetic scoring logic.

## Extensibility
This pipeline is designed to be easily extended:

Add new features such as:

- Token holding duration

- Flash loan usage

- Interest earned or paid

- Replace synthetic scores with real-world loan repayment outcomes if available

- Tune hyperparameters or try alternative models (e.g., XGBoost, LightGBM)

- Convert this into an API or real-time scoring service

- Visualize results using dashboards or integrate with on-chain analytics tools





