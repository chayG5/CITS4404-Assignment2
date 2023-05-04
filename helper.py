# import modules
import ccxt
import pandas as pd
import ta

# gets BTC/AUD OHLCV data for the past 720 days, returns pandas dataframe
def get_OHLCV():
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

# Keeps the individual count
count = 0
# get ohlcv data
ohlcv = get_OHLCV()

# split the data into a test and training set
size = 325
training_data = ohlcv.iloc[:size, :]
test_data = ohlcv.iloc[size:, :]
test_data.reset_index(drop=True, inplace=True)

# get the oHLCV data for the test and training set
test_close = test_data['Close']; test_volume = test_data['Volume']
train_close = training_data['Close']; train_volume = training_data['Volume']
test_high = test_data['High']; test_low = test_data['Low']
train_high = training_data['High']; train_low = training_data['Low']

# get the indicators for the test and training set
rsi_test = ta.momentum.rsi(test_close, window=14); rsi_train = ta.momentum.rsi(train_close, window=14)
macd_test = ta.trend.macd(test_close, 26, 12, 9); macd_train = ta.trend.macd(train_close, 26, 12, 9)
macd_signal_test = ta.trend.macd_signal(test_close, 26, 12, 9); macd_signal_train = ta.trend.macd_signal(train_close, 26, 12, 9)
bbh_test = ta.volatility.bollinger_hband(test_close, window=20, window_dev=2); bbh_train = ta.volatility.bollinger_hband(train_close, window=20, window_dev=2)
obv_test = ta.volume.on_balance_volume(test_close, test_volume); obv_train = ta.volume.on_balance_volume(train_close, train_volume)
stoch_test = ta.momentum.stoch(test_close, test_high, test_low, 14, 3); stoch_train = ta.momentum.stoch(train_close, train_high, train_low, 14, 3)
sma_20_test = ta.trend.sma_indicator(test_close, window=20); sma_20_train = ta.trend.sma_indicator(train_close, window=20)
sma_50_test = ta.trend.sma_indicator(test_close, window=50); sma_50_train = ta.trend.sma_indicator(train_close, window=50)

# a Bool class to restrict inputs for to only true and false (exclude int)
class Bool:
    TRUE = True
    FALSE = False

# buy signal for when the macd crosses the signal line
def comparemacd(t:int) -> Bool:
    if t >= size:
        return macd_test.loc[t] > macd_signal_test.loc[t]
    else:
        return macd_train.loc[t] > macd_signal_train.loc[t]

# rsi when it is below 30
def rsi_30(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] < 30
    else:
        return rsi_train.loc[t] < 30

# rsi when it is above 70
def rsi_70(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] > 70
    else:
        return rsi_train.loc[t] > 70

# When close ohlcv value is above the bollinger band
def detectbbh(t: int) -> Bool:
    if t >= size:
        return test_close.loc[t] > bbh_test.loc[t]
    else:
        return train_close.loc[t] > bbh_train.loc[t]

# volume indicator
def detectObv(t: int) -> Bool:
    if t == 0:
        return False
    if t >= size:
        return obv_test.loc[t] > obv_test.loc[t-1]
    else:
        return obv_train.loc[t] > obv_train.loc[t-1]

# stochastic indicator
def Stoch_20(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] < 20
    else:
        return stoch_train.loc[t] < 20

# stochastic indicator
def Stoch_80(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] > 80
    else:
        return stoch_train.loc[t] > 80

# sma indicator
def sma_20_50(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] > sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] > sma_50_train.loc[t]

# sma indicator
def sma_50_20(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] < sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] < sma_50_train.loc[t]

# input for functions if none available 
def num():
    return 1

