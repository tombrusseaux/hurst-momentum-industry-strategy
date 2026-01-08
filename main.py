 
# HURST × MOMENTUM — INDUSTRY STRATEGY


import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# PARAMÈTRES STRAT (FIXES)

BDD_FILE = "Bdd_Projet_Python.xlsx"
START = "2010-01-01"

from datetime import datetime

END = datetime.today().strftime("%Y-%m-%d")



W_MOM = 60
SKIP = 10
REB_STEP = 21
QUANTILE = 0.30
HURST_WINDOW = 252
MIN_OBS = 200

OUT_FILE = "industry_weights.xlsx"

# DATA LOADING

bdd = pd.read_excel(BDD_FILE).iloc[:, :3]
bdd.columns = ["Ticker", "Industry", "Sector"]

bdd["Ticker"] = (
    bdd["Ticker"]
    .astype(str)
    .str.strip()
    .str.replace(".", "-", regex=False)
)
bdd["Industry"] = bdd["Industry"].astype(str).str.strip()
bdd["Sector"] = bdd["Sector"].astype(str).str.strip()

bdd = bdd.dropna(subset=["Ticker"])
bdd = bdd[bdd["Ticker"] != ""].drop_duplicates(subset=["Ticker"])

tickers = bdd["Ticker"].tolist()
print(f"Tickers: {len(tickers)}")


# FETCH PRICES

data = yf.download(
    tickers,
    start=START,
    end=END,
    auto_adjust=True,
    threads=True,
    progress=False
)

close_prices = data.xs("Close", level=0, axis=1)
close_prices = close_prices.dropna(axis=1, thresh=MIN_OBS)
close_prices = close_prices.sort_index()

bdd = bdd[bdd["Ticker"].isin(close_prices.columns)]
print("Tickers after cleaning:", close_prices.shape[1])


# INDUSTRY RETURNS & INDEX

daily_returns = close_prices.pct_change(fill_method=None)

map_ind = bdd.set_index("Ticker")["Industry"]
ind_returns = daily_returns.rename(columns=map_ind).groupby(axis=1, level=0).mean()

ind_index = (1 + ind_returns).cumprod()
ind_index.iloc[0] = 1.0

print("Nb industries:", ind_index.shape[1])


# SIGNALS

def momentum_with_skip(index_df, W, skip):
    return index_df.shift(skip).pct_change(periods=W - skip, fill_method=None)

def hurst_rs(x):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return np.nan
    y = x - x.mean()
    z = np.cumsum(y)
    R = z.max() - z.min()
    S = y.std(ddof=1)
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(n)

def compute_hurst(index_df, window):
    logret = np.log(index_df).diff()
    h = pd.DataFrame(index=index_df.index, columns=index_df.columns)
    for col in index_df.columns:
        h[col] = logret[col].rolling(window).apply(
            lambda s: hurst_rs(s.values), raw=False
        )
    return h

# Momentum
mom = momentum_with_skip(ind_index, W_MOM, SKIP)

# Hurst
hurst = compute_hurst(ind_index, HURST_WINDOW)
hurst_adj = (hurst - 0.5).clip(lower=0)

# Score 
score = (mom * hurst_adj).shift(1)


# PORTFOLIO WEIGHTS

dates = ind_returns.index
assets = ind_returns.columns

weights = pd.DataFrame(0.0, index=dates, columns=assets)

valid_dates = score.dropna(how="all").index
reb_dates = valid_dates[::REB_STEP]

for i in range(len(reb_dates) - 1):
    d0 = reb_dates[i]
    d1 = reb_dates[i + 1]

    s = score.loc[d0].dropna()
    if s.empty:
        continue

    n = max(1, int(len(s) * QUANTILE))
    longs = s.sort_values(ascending=False).head(n).index

    w = 1.0 / len(longs)
    mask = (dates > d0) & (dates <= d1)
    weights.loc[mask, longs] = w

if len(reb_dates) > 0:
    last_reb = reb_dates[-1]
    last_scores = score.loc[last_reb].dropna()

    if not last_scores.empty:
        n = max(1, int(len(last_scores) * QUANTILE))
        longs = last_scores.sort_values(ascending=False).head(n).index
        w = 1.0 / len(longs)

        mask = dates > last_reb
        weights.loc[mask, longs] = w



# EXPORT EXCEL

weights.to_excel(OUT_FILE)
print(f"Daily industry weights saved to {OUT_FILE}")
