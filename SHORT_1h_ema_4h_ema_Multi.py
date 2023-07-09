import pandas as pd
import numpy as np
import datetime as dt
import time
import ta
import os

from binance.client import Client
from dotenv import load_dotenv

# Get the path to the current directory
current_directory = os.getcwd()

# Specify the path to the .env file relative to the current directory
dotenv_path = os.path.join(current_directory, '.env')

# Load the environment variables from the .env file
load_dotenv(dotenv_path)

api_key = os.getenv('API_KEY')
api_secret = os.getenv('SECRET_KEY')

client = Client(api_key, api_secret, testnet = False)

spot = client.get_account()
s_bal = pd.DataFrame(spot['balances'])
print(s_bal)


timezone = -3
# symbol = 'injusdt'
interval = '1h'

# step between timestamps in milliseconds, 60000 = 1min 
step = 60000 * 3600
start_time = round(time.time() * 1000 - (86400 * 1000 * 10))
end_time = round(time.time() * 1000)

def set_time():
    
    global start_time, end_time
    
    # start epoch till now, use prior 5 days for this strategy
    start_time = round(time.time() * 1000 - (86400 * 1000 * 10))
    end_time = round(time.time() * 1000)


dataframes = {}
def create_raw(symbol, interval_arr, step):
    global start_time, end_time
    
    for interval in interval_arr:
        set_time()
        # Fetch the data using batch requests
        data = []
        while start_time < end_time:
            limit = min(step, end_time - start_time + 1)  # Adjust the limit for the last batch

            while True:
                try:
                    response = client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit, startTime=start_time)
                    break
                except Exception as e:
                    print(f'Error create_raw: {e}')    
                    time.sleep(5)

            if len(response) == 0:
                break  # No more data available, exit the loop
            data.extend(response)
            start_time = response[-1][0] + 1

        # Convert the data to a DataFrame
        columns = [
            "Open_Time", "Open", "High", "Low", "Close", "Volume", "Close_Time",
            "Quote asset volume", "Number of trades", "Taker buy base asset volume",
            "Taker buy quote asset volume", "Ignore"
        ]
        raw_df = pd.DataFrame(data, columns=columns)   

        raw_df = raw_df[['Open_Time', 'Open', 'Close', "High", "Low", 'Volume']]

        dataframes[f'df_{interval}_{symbol}'] = raw_df



interval_arr = ['1h', '4h']
ema_arr = [8, 18, 38]


ep_per = 1.005
sl_per = 1.02
sl_det = 'Close'


def get_klines(symbol, interval):
    global dataframes
    df = dataframes[f'df_{interval}_{symbol}']
    df = df[['Open_Time', 'Open', 'Close', "High", "Low", 'Volume']].astype(float)
    df = df.set_index('Open_Time')

    df.index = pd.to_datetime(df.index, unit='ms') + pd.Timedelta(hours=timezone)
    df = df[~df.index.duplicated(keep='first')]

    return df


symbol_dfs = {}
def multi_timeframes(symbol):
    df = get_klines(symbol, interval_arr[0]).copy()
    df = df.astype(float).round(4)

    for interval in interval_arr:
        raw_df = get_klines(symbol, interval)
        # ema
        for ema in ema_arr:
            column_name = f'ema_{ema}_{interval}'
            df[column_name] = ta.trend.EMAIndicator(raw_df.Close, window=ema, fillna=True).ema_indicator()

        # rsi
    #     rsi = ta.momentum.RSIIndicator(raw_df.Close, window = rsi_int)
    #     df[f'rsi_{interval}'] = rsi.rsi()

        # atr
    #         df['atr'] = ta.volatility.average_true_range(df.High, df.Low, df.Close)


    # reset index and set current index as a column
    df = df.reset_index()

    # set new index with integers
    df = df.set_index(pd.RangeIndex(len(df)))

    # time_format(timezone)
    df['Open_Time'] = df['Open_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # fill up higher time frame empty values with equal interval between each value
    df = df.replace('', np.nan)
    df['ema_8_4h'] = df['ema_8_4h'].interpolate()
    df['ema_18_4h'] = df['ema_18_4h'].interpolate()
    df['ema_38_4h'] = df['ema_38_4h'].interpolate()
    symbol_dfs[f'df_{symbol}'] = df
    
    return symbol_dfs[f'df_{symbol}']


# def check_cross(df, kd_dir):
#     up = df['slow_k'] > df['slow_d']
#     down = df['slow_k'] < df['slow_d']
#     if kd_dir == 'Up':
#         return up.diff() & up
#     if kd_dir == 'Any':
#         return up.diff()
#     if kd_dir == 'Down':
#         return down.diff() & down


# def indicators(df, kd_dir):
def indicators(df):    
    
# bb    
    bb_int = 30
    bb_dev = 2
    bb = ta.volatility.BollingerBands(df['Close'], window=bb_int, window_dev=bb_dev)
    df['bb_u'] = bb.bollinger_hband()
    df['bb_m'] = bb.bollinger_mavg()
    df['bb_l'] = bb.bollinger_lband()  
    
# kd
#     df['slow_k']= ta.momentum.stoch(df['High'], df['Low'], df['Close'], 14, 3)
#     df['slow_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], 14, 3)
    
# kd cross
#     df['kd_cross'] = check_cross(df, kd_dir)



def conditions(df):
    
    df['c1'] = df['ema_8_1h'] <= df['ema_18_1h']
    df['c2'] = df['ema_18_1h'] <= df['ema_38_1h']
    df['c3'] = df['ema_8_4h'] <= df['ema_38_4h']
    df['c4'] = df['Close'] >= df['ema_18_4h'] * ep_per

    # 條件達成
    df['signal'] = df.c1 & df.c2 & df.c3 & df.c4
            
    # 下一根進場
    df['entry'] = False



def cancel_orders(symbol):
    
    open_orders = client.futures_get_open_orders()
    
    if open_orders:
        for order in open_orders:
            if (order['symbol'] == symbol.upper()) & (order['positionSide'] == 'SHORT'):  
                cancel_response = client.futures_cancel_order(symbol=symbol.upper(), orderId=order['orderId'])
    else:
        time.sleep(3)


stop_loss_p = 0
def enter_position(df, symbol):
    global stop_loss_p
        
    #-----Calculate entry price-----#
    close_val = df['Close']

    ema_18_val = df['ema_18_4h']
    
    df.loc[df.index[-1], 'entry_p'] = ema_18_val.loc[close_val.index[-1]] * ep_per
    df.loc[df.index[-1], 'stop_loss'] = ema_18_val.loc[close_val.index[-1]] * sl_per

    #-----position attributes-----#
    usdt_q = 70
    quantity = round(usdt_q / df.loc[df.index[-1], 'Close'], 0)
    entry_p = round(df.loc[df.index[-1], 'entry_p'], 3)
    stop_loss_p = round(df.loc[df.index[-1], 'stop_loss'], 3)
#     take_profit_p = round(df.loc[df.index[-1], 'take_profit'], 3)
    
    cancel_orders(symbol)
    
    # 因為現貨會比期貨價格高，空單經常不會打回進場價，為確保進場，在此作調整
    ticker = client.futures_symbol_ticker(symbol=symbol.upper())

    mark_price = round(float(ticker['price']), 3)

#     future_entry_p = entry_p
#     if entry_p > mark_price:
#         future_entry_p = mark_price
        
    try:
        order = client.futures_create_order(
            symbol=symbol.upper(),
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity=quantity,
            price=mark_price,
            positionSide='SHORT',
        )
            
    except Exception as e:
        print(f'Error create_raw: {e}')   
        time.sleep(1)

    return df



def check_sl(df, symbol, current_k):
    global stop_loss_p

    current_sl = stop_loss_p
    
    # 確定停損
    if (df.iloc[df.index[-2]]['High'] >= current_sl):
                
        cancel_orders(symbol)

        # 如果目前價格高於停損
        ticker = client.futures_symbol_ticker(symbol=symbol.upper())

        mark_price = round(float(ticker['price']), 3)
        
#         stop_price = current_sl
#         if current_sl < mark_price:
#             stop_price = mark_price
        
        try:

            stop_loss_order = client.futures_create_order(
                symbol=symbol.upper(),
                side='BUY',
                type='STOP_MARKET',
                stopPrice=mark_price,
                stopLimitTimeInForce='GTC',
                closePosition = 'true',
                positionSide = 'SHORT'
            )
        
        except Exception as e:
            print(f'Error create_raw: {e}')   
            time.sleep(1)



def check_tp(df, symbol):
    
    try:
        # 隨時偵測出場條件是否成立，不必等 Close 發生
        if ((df.loc[df.index[-1], 'Close'] <= df.loc[df.index[-1], 'bb_l']) or 
            ((df.loc[df.index[-2], 'Volume'] > df.loc[df.index[-3], 'Volume']) & # vol greater
            (df.loc[df.index[-3], 'Open'] > df.loc[df.index[-3], 'Close']) & # previous red candle
            (df.loc[df.index[-2], 'Open'] < df.loc[df.index[-2], 'Close']) & # current green candle
            (df.loc[df.index[-2], 'High'] > df.loc[df.index[-3], 'Open'])) # sign of engulf
           ):
            
            cancel_orders(symbol)

            ticker = client.futures_symbol_ticker(symbol=symbol.upper())

            mark_price = round(float(ticker['price']), 3)
                
            take_profit_order = client.futures_create_order(
                symbol=symbol.upper(),
                side='BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=mark_price,
                stopLimitTimeInForce='GTC',
                closePosition = 'true',
                positionSide = 'SHORT'
            )

    except Exception as e:
        print(f'Error create_raw: {e}')   
        time.sleep(2)



current_bar_pos = False
def check_price(df, symbol):
    global current_k, current_bar_pos
    
    try:
        if ((df.iloc[df.index[-1]]['signal']) & # 三線條件成立
            (current_bar_pos == False)): # 本 K 未進場
            enter_position(df, symbol)
            df.loc[df.index[-1], 'entry'] = True
            current_bar_pos = True
            
        # 如果時間標籤改變，設 current_bar_pos 為可進場
        if df['Open_Time'][len(df) - 1] != current_k:

            current_k = df['Open_Time'][len(df) - 1]
            current_bar_pos = False
            
    except Exception as e:
        print(f'Error create_raw: {e}')   
        time.sleep(1)


loop_start_time = 0
loop_end_time = 0

line_count = 0
max_lines = 1
current_k = 0
restart_countdown = 2 * 60 * 60 # restart in 2 hours

def run(symbol):
    global current_k, line_count, max_lines, restart_countdown, loop_start_time, loop_end_time, stop_loss_p

    while True:
        # console_df = {}
        while True:
            try:
                create_raw(symbol, interval_arr, step)
                break
            except ConnectionError as e:
                print("Connection error occurred:", e)
                print("Retrying in 5 seconds...")
                time.sleep(5)
                
        try:
            set_time()
            df = multi_timeframes(symbol)
#                 indicators(df, kd_dir)
            indicators(df)
            conditions(df)
            check_price(df, symbol)   
            # print(df.tail(1))
            df.to_csv(f'SHORT_1h_ema_4h_ema_Multi.csv')
            # 檢查停損
            try:
                positions_info = client.futures_account()['positions']
                short_positions = [p for p in positions_info if p['positionSide'] == 'SHORT' and float(p['positionAmt']) != 0 and p['symbol'] == symbol.upper()]

                if short_positions:
                    check_tp(df, symbol)
                    check_sl(df, symbol, current_k)

                else:
                    stop_loss_p = 0

            except Exception as e:
                time.sleep(3) 

                
        except ConnectionError as e:
            print("Connection error occurred:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
