# import modules
import ccxt
import pandas as pd
import ta


def get_OHLCV():
    # downloads BTC/AUD OHLCV data for the past 720 days, returns pandas dataframe
    exch = "kraken"
    time_frame = "1d"
    symbol = "BTC/AUD"
    exchange = getattr(ccxt, exch)()
    data = pd.DataFrame(
        exchange.fetchOHLCV(symbol=symbol, timeframe=time_frame),
        columns=["Date", "Open", "High", "Low", "Close", "Volume"],
    )

    # convert unix timestamp to readable date
    data["Date"] = pd.to_datetime(data["Date"], unit="ms")
    return data


# data = get_OHLCV().to_csv("OHLCV_data")


def add_taIndicators():
    ohlcv = get_OHLCV()
    # Add Bollinger Bands to the DataFrame
    indicator_bb = ta.volatility.BollingerBands(
        close=ohlcv["Close"], window=20, window_dev=2
    )
    ohlcv["bb_high"] = indicator_bb.bollinger_hband()
    ohlcv["bb_low"] = indicator_bb.bollinger_lband()
    # Add RSI to the DataFrame
    indicator_rsi = ta.momentum.RSIIndicator(close=ohlcv["Close"], window=14)
    ohlcv["rsi"] = indicator_rsi.rsi()

    # Drop rows with missing values
    ohlcv.dropna(inplace=True)
    return ohlcv


# data = add_taIndicators().to_csv("OHLCV_data")
