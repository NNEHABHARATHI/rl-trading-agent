"""
RL Trading Agent (PPO) vs Buy-and-Hold vs SMA crossover
Install: pip install yfinance pandas numpy gymnasium stable-baselines3 matplotlib
"""
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

TICKER = "SPY"          # NIFTY ki "^NSEI" use cheyyandi
START, END = "2012-01-01", "2024-12-31"
WINDOW = 30
COST = 0.001            # 0.1% transaction cost
TRAIN_FRAC = 0.7
SEED = 42


# ---------- 1. Data + features ----------
def load_data():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()

    c = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["ret"] = c.pct_change()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    out["macd"] = (c.ewm(span=12).mean() - c.ewm(span=26).mean()) / c
    out["vol20"] = out["ret"].rolling(20).std()
    out["vol_chg"] = df["Volume"].pct_change().clip(-1, 1)
    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["close"] = c
    return out.dropna()


FEATS = ["ret", "rsi", "macd", "vol20", "vol_chg"]


# ---------- 2. Environment ----------
class TradingEnv(gym.Env):
    """Actions: 0=Hold, 1=Buy (go long), 2=Sell (go flat)."""

    def __init__(self, features, returns, window=WINDOW, cost=COST):
        super().__init__()
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window, self.cost = window, cost
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(window * self.X.shape[1] + 1,), dtype=np.float32
        )

    def _obs(self):
        past = self.X[self.t - self.window:self.t]  # sirf past data (no leakage)
        return np.append(past.flatten(), self.pos).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.pos = self.window, 0
        return self._obs(), {}

    def step(self, action):
        new_pos = 1 if action == 1 else (0 if action == 2 else self.pos)
        reward = new_pos * self.r[self.t] - self.cost * abs(new_pos - self.pos)
        self.pos = new_pos
        self.t += 1
        done = self.t >= len(self.r)
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), done, False, {"net_ret": float(reward)}


# ---------- 3. Metrics ----------
def metrics(daily):
    daily = np.asarray(daily)
    equity = np.cumprod(1 + daily)
    sharpe = daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)
    mdd = (equity / np.maximum.accumulate(equity) - 1).min()
    return {
        "Total Return %": round((equity[-1] - 1) * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown %": round(mdd * 100, 2),
        "Win Rate %": round((daily > 0).mean() * 100, 2),
    }


def run_agent(model, env):
    obs, _ = env.reset()
    rets, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        rets.append(info["net_ret"])
    return np.array(rets)


# ---------- 4. Main ----------
def main():
    df = load_data()
    split = int(len(df) * TRAIN_FRAC)
    train, test = df.iloc[:split], df.iloc[split:]

    # Normalize using TRAIN stats only
    mu, sd = train[FEATS].mean(), train[FEATS].std() + 1e-9
    Xtr = ((train[FEATS] - mu) / sd).values
    Xte = ((test[FEATS] - mu) / sd).values

    train_env = TradingEnv(Xtr, train["ret"].values)
    test_env = TradingEnv(Xte, test["ret"].values)

    model = PPO("MlpPolicy", train_env, verbose=1, seed=SEED,
                learning_rate=3e-4, n_steps=1024, batch_size=64, ent_coef=0.01)
    model.learn(total_timesteps=200_000)

    # Agent
    agent_rets = run_agent(model, test_env)

    # Baselines (same period as agent: window nundi)
    test_ret = test["ret"].values[WINDOW:]
    bh = test_ret
    sig = (test["sma20"] > test["sma50"]).astype(int).shift(1).fillna(0).values[WINDOW:]
    sma = sig * test_ret - COST * np.abs(np.diff(sig, prepend=0))

    results = pd.DataFrame({
        "PPO Agent": metrics(agent_rets),
        "Buy & Hold": metrics(bh),
        "SMA Crossover": metrics(sma),
    }).T
    print("\n=== Out-of-sample results ===")
    print(results)
    results.to_csv("results.csv")

    # Equity curves
    plt.figure(figsize=(10, 5))
    for name, r in [("PPO Agent", agent_rets), ("Buy & Hold", bh), ("SMA Crossover", sma)]:
        plt.plot(np.cumprod(1 + r), label=name)
    plt.title(f"Out-of-sample equity curve ({TICKER})")
    plt.xlabel("Trading days")
    plt.ylabel("Growth of 1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
