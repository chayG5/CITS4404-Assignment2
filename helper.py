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

count = 0
ohlcv = get_OHLCV()
size = 325

training_data = ohlcv.iloc[:size, :]
test_data = ohlcv.iloc[size:, :]
test_data.reset_index(drop=True, inplace=True)

test_close = test_data['Close']; test_volume = test_data['Volume']
train_close = training_data['Close']; train_volume = training_data['Volume']
test_high = test_data['High']; test_low = test_data['Low']
train_high = training_data['High']; train_low = training_data['Low']

rsi_test = ta.momentum.rsi(test_close, window=14)
rsi_train = ta.momentum.rsi(train_close, window=14)
macd_test = ta.trend.macd(test_close, 26, 12, 9)
macd_train = ta.trend.macd(train_close, 26, 12, 9)
macd_signal_test = ta.trend.macd_signal(test_close, 26, 12, 9)
macd_signal_train = ta.trend.macd_signal(train_close, 26, 12, 9)
bbh_test = ta.volatility.bollinger_hband(test_close, window=20, window_dev=2)
bbh_train = ta.volatility.bollinger_hband(train_close, window=20, window_dev=2)
obv_test = ta.volume.on_balance_volume(test_close, test_volume)
obv_train = ta.volume.on_balance_volume(train_close, train_volume)
stoch_test = ta.momentum.stoch(test_close, test_high, test_low, 14, 3)
stoch_train = ta.momentum.stoch(train_close, train_high, train_low, 14, 3)
sma_20_test = ta.trend.sma_indicator(test_close, window=20)
sma_20_train = ta.trend.sma_indicator(train_close, window=20)
sma_50_test = ta.trend.sma_indicator(test_close, window=50)
sma_50_train = ta.trend.sma_indicator(train_close, window=50)

class Bool:
    TRUE = True
    FALSE = False

class Constant:
    one = 0.1
    two = 0.2
    three = 0.3
    four = 0.4
    five = 0.5
    actualOne = 1

def comparemacd(t:int) -> Bool:
    if t >= size:
        return macd_test.loc[t] > macd_signal_test.loc[t]
    else:
        return macd_train.loc[t] > macd_signal_train.loc[t]

def rsi_30(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] < 30
    else:
        return rsi_train.loc[t] < 30

def rsi_70(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] > 70
    else:
        return rsi_train.loc[t] > 70


def detectbbh(t: int) -> Bool:
    if t >= size:
        return test_close.loc[t] > bbh_test.loc[t]
    else:
        return train_close.loc[t] > bbh_train.loc[t]

def detectObv(t: int) -> Bool:
    if t == 0:
        return False
    if t >= size:
        return obv_test.loc[t] > obv_test.loc[t-1]
    else:
        return obv_train.loc[t] > obv_train.loc[t-1]

def Stoch_20(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] < 20
    else:
        return stoch_train.loc[t] < 20

def Stoch_80(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] > 80
    else:
        return stoch_train.loc[t] > 80

def sma_20_50(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] > sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] > sma_50_train.loc[t]

def sma_50_20(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] < sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] < sma_50_train.loc[t]

def num():
    return 1

