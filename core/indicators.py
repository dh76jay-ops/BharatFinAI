import pandas as pd
import numpy as np
import pandas_ta as ta

def calc_indicators(df):
    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["Close"], length=20)
    df["BB_Upper"] = bb.iloc[:, 0]
    df["BB_Lower"] = bb.iloc[:, 2]

    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)

    df["Vol_MA"] = df["Volume"].rolling(20).mean()

    return df
