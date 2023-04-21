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
    ohlcv['SMA'] = ta.trend.sma_indicator(ohlcv['close'], window=20)
    ohlcv['RSI'] = ta.momentum.rsi_indicator(ohlcv['close'], window=14)
    ohlcv['MACD'] = ta.trend.macd(ohlcv['close'], window_slow=26, window_fast=12, window_sign=9)['MACD']
    ohlcv['BB_lower'] = ta.volatility.bollinger_lband_indicator(ohlcv['close'], window=20)
    ohlcv['BB_upper'] = ta.volatility.bollinger_hband_indicator(ohlcv['close'], window=20)
    ohlcv['ADX'] = ta.trend.adx_indicator(ohlcv['high'], ohlcv['low'], ohlcv['close'], window=14)
    ohlcv['OBV'] = ta.volume.on_balance_volume(ohlcv['close'], ohlcv['volume'])
    ohlcv[['Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou']] = ta.trend.IchimokuIndicator(ohlcv['high'], ohlcv['low'], conversion_line_window=9, base_line_window=26, lagging_span2_window=52).trend_indicator()

    # Drop rows with missing values
    ohlcv.dropna(inplace=True)
    return ohlcv

