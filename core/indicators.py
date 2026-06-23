import pandas as pd
import numpy as np

def calc_indicators(df):
    df = df.copy()

    close = df["Close"]

    # SMA / EMA
    df["SMA_50"] = close.rolling(window=50).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Bollinger Bands
    df["BB_MID"] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_UPPER"] = df["BB_MID"] + (2 * bb_std)
    df["BB_LOWER"] = df["BB_MID"] - (2 * bb_std)

    return df