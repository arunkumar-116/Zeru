# 🏦 DeFi Wallet Credit Scoring

This project assigns a **credit score (0–1000)** to DeFi wallets based on historical transaction behavior using Aave V2 data. It uses feature engineering and a machine learning model to quantify creditworthiness for each wallet.

---

## 📂 Input

- **File**: `user-wallet-transactions.json`
- **Format**: List of transaction records from Aave V2
- **Key Fields**:
  - `userWallet`
  - `action`
  - `timestamp`
  - `actionData.assetSymbol`
  - `actionData.amount`
  - `actionData.assetPriceUSD`

---

## 🧠 Feature Engineering

The following wallet-level features are computed:

| Feature             | Description                               |
|---------------------|-------------------------------------------|
| `tx_count`          | Total number of transactions              |
| `deposit_usd`       | Total USD value deposited                 |
| `redeem_usd`        | Total USD value redeemed                  |
| `borrow_usd`        | Total USD value borrowed                  |
| `repay_usd`         | Total USD value repaid                    |
| `liquidation_count` | Number of liquidation calls               |
| `active_days`       | Days between first and last activity      |
| `unique_tokens`     | Count of unique tokens used               |
| `tx_freq`           | Transactions per active day               |

### 🔁 Token Normalization

Token amounts are normalized using predefined decimal mappings (e.g., USDC = 6, WBTC = 8).

---

## 🎯 Synthetic Score Logic

For training, a synthetic score is generated using a rule-based formula:

```python
score = (
    + 0.4 * repay_usd
    - 0.3 * borrow_usd
    + 0.2 * deposit_usd
    - 100 * liquidation_count
    + 100
)

