# RL Trading Agent (PPO) vs Buy-and-Hold

A reinforcement learning trading agent built with a custom Gymnasium environment and Stable-Baselines3 PPO, evaluated out-of-sample against Buy-and-Hold and an SMA crossover baseline.

## Goal

Test whether an RL agent can learn a trading policy that beats simple baselines after accounting for transaction costs, using strict train/test separation to avoid look-ahead bias.

## Approach

*Data*: Daily OHLCV for SPY (2012-2024) via yfinance.

*Features*: Daily returns, RSI(14), MACD (normalized), 20-day rolling volatility, volume change. Normalized using training-set statistics only.

*Environment* (custom Gymnasium env):
- *State*: last 30 days of features + current position
- *Actions*: Hold / Buy (go long) / Sell (go flat)
- *Reward*: position x next-day return - transaction cost
- *Transaction cost*: 0.1% per position change

*Algorithm*: PPO (Stable-Baselines3, MLP policy), 200k timesteps.

*Evaluation*: Chronological split, first 70% for training and last 30% for testing (no random shuffling, no leakage).

*Baselines*: Buy-and-Hold, SMA(20/50) crossover (with the same transaction costs).

## Results (out-of-sample)

| Strategy | Total Return % | Sharpe | Max Drawdown % | Win Rate % |
|---|---|---|---|---|
| PPO Agent | 18.95 | 0.41 | -19.31 | 33.87 |
| Buy & Hold | 50.81 | 0.75 | -24.50 | 53.74 |
| SMA Crossover | 3.81 | 0.14 | -29.60 | 36.32 |

### Equity curve



![Equity curve](equity_curve.png)



### Analysis

The PPO agent did not beat Buy-and-Hold on total return or Sharpe over the test period. This is expected in a strongly trending market, since a long-only strategy that exits positions pays transaction costs and misses upside. It did, however, reduce maximum drawdown (-19.3% vs -24.5%) and clearly outperformed the SMA crossover baseline (18.95% vs 3.81% return, Sharpe 0.41 vs 0.14), showing it learned a useful risk-managing policy. Results are from a single seed and a single split, so they should be read as indicative, not conclusive.

## Run it

bash
git clone https://github.com/NNEHABHARATHI/rl-trading-agent.git
cd rl-trading-agent
pip install -r requirements.txt
python rl_trading_agent.py


Outputs: results.csv and equity_curve.png.

To change the asset, edit TICKER in rl_trading_agent.py (e.g. "^NSEI" for NIFTY 50).

## Limitations

- Daily data and a simple long/flat strategy; no shorting or position sizing.
- Single asset, single train/test split, single seed. Results can vary across seeds.
- Costs are modeled as a flat percentage; no slippage or market impact.
- Past performance does not imply future results. This is an educational project, not financial advice.

## Future work

- Walk-forward validation across multiple windows
- LSTM/Transformer price-prediction signal as an additional RL state feature
- Multi-asset portfolio allocation
- Multiple seeds with confidence intervals

## Tech stack

Python,
