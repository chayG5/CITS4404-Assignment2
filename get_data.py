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
    # calculate the technical indicators using the ta library
    ohlcv['SMA'] = ta.trend.sma_indicator(ohlcv['Close'], window=20)
    ohlcv['RSI'] = ta.momentum.rsi(ohlcv['Close'], window=14)
    ohlcv['BB_lower'] = ta.volatility.bollinger_lband(ohlcv['Close'], window=20)
    ohlcv['BB_upper'] = ta.volatility.bollinger_hband(ohlcv['Close'], window=20)

    # Drop rows with missing values
    ohlcv.dropna(inplace=True)
    return ohlcv
