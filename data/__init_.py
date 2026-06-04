import yfinance as yf

def get_data(symbol, period):

    s = symbol.upper().strip()

    if not s.endswith(".NS") and not s.endswith(".BO"):
        s = s + ".NS"

    ticker = yf.Ticker(s)

    df = ticker.history(period=period)

    info = ticker.info

    return df, info, s