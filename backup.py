#!/usr/bin/env python
# coding: utf-8

# In[1]:


# k crossover d，close > ema 8, ema 8 > ema 18, ema 18 > 38, take profit atr, stop loss atr
# Variables :
# time - 15m, 1h
# start time - 1609492611000, 1641028611000, 1672564611000 (2021, 2022, 2023)
# tp atr - 4, 6
# sl atr - 3, 5
# sl - Close, Low


# In[2]:


# ! conda install -c conda-forge ta --yes


# In[3]:


import requests
import pandas as pd
import ta
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import time
import itertools
import multiprocessing as mp


# In[4]:


start_time_arr = [1672535048000, 1640999048000, 1609463048000]
interval_arr = ['15m', '1h']
kd_direction_arr = ['Up', 'Any', 'Down']
sl_atr_arr = [3, 5]
tp_atr_arr = [4, 6]
sl_determine_arr = ['Close', 'Low']


# In[5]:


timezone = 8
endpoint = 'wss://stream.binance.com:9443/ws'
symbol = 'ethusdt'
symbol_C = symbol.upper()

end_time = round(time.time() * 1000)

# step between timestamps in milliseconds
step = 60000 * 3600


# In[6]:


def create_raw(symbol, interval_arr, start_time, end_time, step):
    
    url = "https://api.binance.com/api/v3/klines"
    
    for interval in interval_arr:

        raw_df = pd.DataFrame()
        
        for timestamp in range(start_time, end_time, step):
            params = {"symbol": symbol_C,
                      "interval": interval,
                      "startTime": timestamp,
                      "endTime": timestamp + step}
            response = requests.get(url, params=params).json()
            out = pd.DataFrame(response, columns = ["Open time", "Open", "High", "Low", "Close",
                                                   "Volume", "Close_Time", "Quote asset volume",
                                                   "Number of trades", "Taker buy base asset volume",
                                                   "Taker buy quote asset volume", "Ignore"])
            raw_df = pd.concat([raw_df, out], axis = 0)

        raw_df = raw_df[['Close_Time', 'Open', 'Close', "High", "Low", 'Volume']]

        raw_df.to_hdf(f'klines_{symbol}_{interval}.h5', key='df', mode='w')
    


# In[7]:


create_raw(symbol, interval_arr, min(start_time_arr), end_time, step)


# In[8]:


def get_klines(symbol, interval, start_time, end_time, step):
    
    df = pd.read_hdf(f'klines_{symbol}_{interval}.h5', key='df')
    
    df = df[(df['Close_Time'] >= start_time) & (df['Close_Time'] <= end_time)]
    
    df = df[['Close_Time', 'Open', 'Close', "High", "Low", 'Volume']]
    convert_dict = {'Close_Time': float, 'Open': float, 'Close': float, "High": float, "Low": float, 'Volume': float}
    df = df.astype(convert_dict)
    
    df['Close_Time'] = pd.to_datetime(df['Close_Time'], unit = 'ms')
    df['Close_Time'] = df['Close_Time'] + pd.Timedelta(hours=timezone)
    df['Close_Time'] = df['Close_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df = df.reset_index(drop=True)
    
    return df


# In[9]:


def check_cross(df, kd_direction):
    up = df['slow_k'] > df['slow_d']
    down = df['slow_k'] < df['slow_d']
    if kd_direction == 'Up':
        return up.diff() & up
    if kd_direction == 'Any':
        return up.diff()
    if kd_direction == 'Down':
        return down.diff() & down


def indicators(df, kd_direction):

# ema
    for i in (8, 18, 38):
        df['ema_'+str(i)] = ta.trend.ema_indicator(df.Close, window=i)

# atr
    df['atr'] = ta.volatility.average_true_range(df.High, df.Low, df.Close)
    
# rsi
    rsi_int = 14
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window = rsi_int).rsi()

# kd
    kd_int = 14
    d_int = 3
   
    kddf = pd.DataFrame()
    kddf[str(kd_int) + '-Low'] = df['Low'].rolling(kd_int).min()
    kddf[str(kd_int) + '-High'] = df['High'].rolling(kd_int).max()
    df['slow_k'] = (df['Close'] - kddf[str(kd_int) + '-Low'])*100/(kddf[str(kd_int) + '-High'] - kddf[str(kd_int) + '-Low'])
    df['slow_d'] = df['slow_k'].rolling(d_int).mean()
    
# kd cross
    df['kd_cross'] = check_cross(df, kd_direction)
    
    return df


# In[10]:


def conditions(df):

    for index, row in df.iterrows():
        # c1
        df['c1'] = df['kd_cross']
        # c2
        df['c2'] = df['Close'] >= df['ema_8']
        # c3
        df['c3'] = df['ema_8'] >= df['ema_18']
        # c4
        df['c4'] = df['ema_18'] >= df['ema_38']

    # 條件達成
    df['signal'] = False
    df.loc[df.c1 & df.c2 & df.c3 & df.c4 , 'signal'] = True


    # 下一根進場
    df['open_entry'] = False
    for i in range(len(df) - 1):
        if df.loc[i, 'signal'] == True:
            df.loc[i + 1, 'open_entry'] = True
    
    return df


# In[11]:


def entries(df, sl_atr, tp_atr, sl_determine):

    in_position = False
    stop_loss = np.nan
    take_profit = np.nan
    close_val = df['Close']
    atr_val = df['atr']

    for index, row in df.iterrows():

        if index == 0:
            continue

        elif df.at[index, 'open_entry'] == True:

            df.at[index, 'entry_p'] = close_val.shift(1).at[index]
            df.at[index, 'stop_loss'] = close_val.shift(1).at[index] - sl_atr * atr_val.shift(1).at[index]
            df.at[index, 'take_profit'] = close_val.shift(1).at[index] + tp_atr * atr_val.shift(1).at[index]
            df.at[index, 'position'] = 'Buy'
            in_position = True
            stop_loss = df.at[index, 'stop_loss']
            take_profit = df.at[index, 'take_profit']


        # 吃筍
        #-----------------------------重要-----------------------------
        # 若用 if 寫，則有可能入場馬上吃筍，若用 elif 則一個 iteration 只會執行一次
        elif in_position == True and (df.at[index, sl_determine] <= stop_loss):
            df.at[index, 'position'] = 'Stop'
            in_position = False
            stop_loss = np.nan
            take_profit = np.nan

        # set take profit
        elif in_position == True and (df.at[index, 'High'] >= take_profit):
            df.at[index, 'position'] = 'Sell'
            in_position = False
            stop_loss = np.nan
            take_profit = np.nan
    

    # 過濾有訊號或事件發生的Ｋ線
    df = df[(df['open_entry'] == True) |
                  (df['signal'] == True) | 
                  (df['position'] == 'Buy') |
                  (df['position'] == 'Sell') |
                  (df['position'] == 'Stop')]


# In[12]:


# 部位回測
def backtest(df):

    df = df.reset_index(drop = True)
    df = df[(df['position'] == 'Buy') |
                  (df['position'] == 'Sell') |
                  (df['position'] == 'Stop')]

    # 一次進場多少單位
    pos_size = 1

    col = ['Close_Time', 'Open', 'Close', 'High', 'Low', 'ema_8', 'ema_18', 'ema_38', 'atr', 'kd_cross', 'position','entry_p', 'stop_loss', 'take_profit']
    pos = df[col]
    pos = pos.reset_index(drop = True)


    for index, row in pos.iterrows():

        current_pos = 0

        # 進場
        if pos.at[index, 'position'] == 'Buy':
            pos.at[index, 'size'] = pos_size
            pos.exit_p = np.nan

        # 出場
        if pos.at[index, 'position'] == 'Sell' or pos.at[index, 'position'] == 'Stop':

            # 停利：達成條件時收盤價
            if pos.at[index, 'position'] == 'Sell':
                for i in range(index -1, -1, -1):
                    if pos.at[i, 'position'] == 'Buy':
                        pos.at[index, 'exit_p'] = pos.at[i, 'take_profit']
                    break

            # 停損：打到進場停損點（往回跌代，直到最近的'Buy'及其'stop_loss'）
            if pos.at[index, 'position'] == 'Stop':
                for i in range(index -1, -1, -1):
                    if pos.at[i, 'position'] == 'Buy':
                        pos.at[index, 'exit_p'] = pos.at[i, 'stop_loss']
                    break

            # 計算每次出場部位大小（每次出場皆清倉）
            for i in range(index -1, -1, -1):
                if pos.at[i, 'position'] == 'Buy':
                    current_pos += pos.at[i, 'size']
                    if i == 0:
                        pos.at[index, 'size'] = -current_pos
                    else:
                        continue
                else:
                    pos.at[index, 'size'] = -current_pos
                    current_pos = 0
                    break


    # 計算部位價值
    for index, row in pos.iterrows():
        if pos.at[index, 'position'] == 'Buy':
            pos.at[index, 'amt'] = round(pos.at[index, 'size'] * pos.at[index, 'entry_p'], 4)
        elif pos.at[index, 'position'] == 'Sell' or pos.at[index, 'position'] == 'Stop':
            pos.at[index, 'amt'] = round(pos.at[index, 'size'] * pos.at[index, 'exit_p'], 4)


    # 若最後一筆為 Buy，移除該單，迭代驗證
    for index, row in pos.iloc[::-1].iterrows():
        if row['position'] == 'Buy':
            pos = pos.drop(index)
        else:
            break


    # 手續費、滑點、價差
    fee = 0.05 / 100
    amt_abs_sum = pos.amt.abs().sum()
    ttl_fee = amt_abs_sum * fee

    # 損益
    leverage = 10
    ttl_profit = -pos.amt.sum() - ttl_fee


    # 計算進場最大部位，最大損益
    consec_entry = 0
    position_amt_sum = 0
    max_consec_entry = 0
    max_position = 0
    max_profit = 0
    max_loss = 0

    for index, row in pos.iterrows():

        if row['position'] == 'Buy':

            consec_entry += 1
            position_amt_sum += row['amt']

        elif row['position'] in ['Sell', 'Stop']:

            if consec_entry > max_consec_entry:
                max_consec_entry = consec_entry
                max_position = position_amt_sum

            position_amt_sum += row['amt']

            if -position_amt_sum > max_profit:
                max_profit = -position_amt_sum

            if -position_amt_sum < max_loss:
                max_loss = -position_amt_sum

            consec_entry = 0
            position_amt_sum = 0

        else:
            pass


    profit_per = "{:.2f}%".format(ttl_profit / (max_position/leverage) * 100)

    wins = pos['position'].str.count('Sell').sum()
    loses = pos['position'].str.count('Stop').sum()

    win_rate = "{:.2f}%".format(wins / (wins + loses) * 100)

    result = {'Profit': [round(ttl_profit, 2)],
              'Fee': [round(ttl_fee, 2)],
              'Max_Profit': [round(max_profit, 2)],
              'Max_Loss': [round(max_loss, 2)],
              'Max_Entry': [max_consec_entry],
              'Max_Position': [round(max_position, 2)],
              'Profit_%': [profit_per],
              'Win_Rate': [win_rate]}


    result_df = pd.DataFrame(result)

    return result_df


# In[ ]:


def backtest_all(symbol, start_time, end_time, step, interval_arr, kd_direction_arr, sl_atr_arr, tp_atr_arr, sl_determine_arr):
    result = pd.DataFrame()

    for interval in interval_arr:
        df = get_klines(symbol, interval, start_time, end_time, step)

        for variables in itertools.product(kd_direction_arr, sl_atr_arr, tp_atr_arr, sl_determine_arr):
            kd_direction, sl_atr, tp_atr, sl_determine = variables

            indicators(df, kd_direction)
            conditions(df)
            entries(df, sl_atr, tp_atr, sl_determine)

            result = pd.concat([result, backtest(df)], ignore_index=True)

    return result


# In[32]:


from backtester import backtest_all
if __name__ == "__main__":
    
    print("Starting multiprocessing...")

    with mp.Pool() as pool:

        results = pool.starmap(backtest_all, [(start_time, interval_arr, kd_direction_arr, sl_atr_arr, tp_atr_arr, sl_determine_arr) for start_time in start_time_arr])
        
    print("Multiprocessing finished.")

    results_df = pd.concat(results, ignore_index=True)

    results_df['start_time'] = np.repeat(start_time_arr, len(interval_arr) * len(kd_direction_arr) * len(sl_atr_arr) * len(tp_atr_arr) * len(sl_determine_arr))
    results_df['interval'] = np.tile(interval_arr, len(start_time_arr) * len(kd_direction_arr) * len(sl_atr_arr) * len(tp_atr_arr) * len(sl_determine_arr))
    results_df['kd_direction'] = np.tile(np.repeat(kd_direction_arr, len(interval_arr)), len(start_time_arr) * len(sl_atr_arr) * len(tp_atr_arr) * len(sl_determine_arr))
    results_df['sl_atr'] = np.tile(np.repeat(sl_atr_arr, len(kd_direction_arr) * len(interval_arr)), len(start_time_arr) * len(tp_atr_arr) * len(sl_determine_arr))
    results_df['tp_atr'] = np.tile(np.repeat(tp_atr_arr, len(sl_atr_arr) * len(kd_direction_arr) * len(interval_arr)), len(start_time_arr) * len(sl_determine_arr))
    results_df['sl_determine'] = np.tile(sl_determine_arr, len(start_time_arr) * len(interval_arr) * len(kd_direction_arr) * len(sl_atr_arr) * len(tp_atr_arr))

    results_df = results_df[['start_time', 'interval', 'kd_direction',
                             'sl_atr', 'tp_atr', 'sl_determine',
                             'Profit', 'Fee', 'Max_Profit', 'Max_Loss',
                             'Max_Entry', 'Max_Position', 'Profit_%']]


# In[ ]:


results_df = results_df.sort_values('Profit', ascending = False)
print(results_df)


# In[ ]:


results_df.to_csv('results_df.csv')

