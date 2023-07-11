import pandas as pd
import numpy as np
import datetime as dt
import time
import ta
import os

# import winsound
# frequency = 250  # Set Frequency To 2500 Hertz
# duration = 300  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)

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
interval = '1h'
# symbol = 'btcusdt'

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
                except:
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


ep_per = 1.003
sl_per = 0.99
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
    df = get_klines(symbol, interval_arr[0]).copy() # eth 1h
    df = df.astype(float).round(4)

    for interval in interval_arr:
        raw_df = get_klines(symbol, interval)
        # ema
        for ema in ema_arr:
            column_name = f'ema_{ema}_{interval}' # eth 1h 4th ema
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
    
    df['c1'] = df['ema_8_1h'] >= df['ema_18_1h']
    df['c2'] = df['ema_18_1h'] >= df['ema_38_1h']
    df['c3'] = df['ema_8_4h'] >= df['ema_38_4h']
    df['c4'] = df['Close'] <= df['ema_18_4h'] * ep_per

    # 條件達成
    df['signal'] = df.c1 & df.c2 & df.c3 & df.c4
    
    # 下一根進場
    df['entry'] = False



def cancel_orders(symbol):
    
    open_orders = client.futures_get_open_orders()
    
    # Cancel all open orders
    if open_orders:
        # Cancel all open orders
        for order in open_orders:
            if (order['symbol'] == symbol.upper()) & (order['positionSide'] == 'LONG'):  
                cancel_response = client.futures_cancel_order(symbol=symbol.upper(), orderId=order['orderId'])
                # print(f"Canceled order: {order['symbol']} - {order['orderId']}")
    else:
        # Continue with the rest of the code
        # print("No open LONG orders found.")
        time.sleep(1)  # Sleep for 1 second to avoid API rate limit


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

    try:
        order = client.futures_create_order(
            symbol=symbol.upper(),
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity=quantity,
            price=entry_p,
            positionSide='LONG',
        )
        # print('Order created successfully.')       
            
    except Exception as e:
        print(f'Error creating order: {e}')
        
    # print(str(symbol.upper()) + ' entered at ' + str(entry_p))

    return df


def check_sl(df, symbol, current_k):
    global stop_loss_p
    # 檢查前一根 Close 是否低於 SL，若低於，即刻停損
    
#     last_signal_row = df.loc[df['signal'].eq(True) & df['Open_Time'].ne(current_k)].tail(1)
#     if not last_signal_row.empty:
#         current_sl = round(df.loc[last_signal_row.index[0], 'Low'] * sl_per, 3)
#         current_tp = round(df.loc[last_signal_row.index[0], 'Close'] + df.loc[last_signal_row.index[0], 'atr'] * tp_atr, 3)

    current_sl = stop_loss_p
    print(f'Current SL at {current_sl}')
    print(f'Current TP at DYNAMIC')
    
    # 確定停損
    if (df.iloc[df.index[-2]]['Low'] <= current_sl):
        
        # print('Last candle closed below SL, try create SL Order')
        
        cancel_orders(symbol)

        # 如果目前價格高於停損
        ticker = client.futures_symbol_ticker(symbol=symbol.upper())

        mark_price = round(float(ticker['price']), 3)
        
#         stop_price = current_sl
#         if current_sl > mark_price:
#             stop_price = mark_price
        
        try:

            stop_loss_order = client.futures_create_order(
                symbol=symbol.upper(),
                side='SELL',
                type='STOP_MARKET',
                stopPrice=mark_price,
                stopLimitTimeInForce='GTC',
                closePosition = 'true',
                positionSide = 'LONG'
            )
        
            # print(f'Stop loss created successfully at: {current_sl}')
            # winsound.Beep(frequency, duration)
        
        except Exception as e:
            
            print(f'Error creating SL order: {e}')
          

def check_tp(df, symbol):
    
    try:
        # 隨時偵測出場條件是否成立，不必等 Close 發生
        if ((df.loc[df.index[-1], 'Close'] >= df.loc[df.index[-1], 'bb_u']) or 
            ((df.loc[df.index[-2], 'Volume'] > df.loc[df.index[-3], 'Volume']) & # vol greater
            (df.loc[df.index[-3], 'Open'] < df.loc[df.index[-3], 'Close']) & # previous green candle
            (df.loc[df.index[-2], 'Open'] > df.loc[df.index[-2], 'Close']) & # current red candle
            (df.loc[df.index[-2], 'Low'] < df.loc[df.index[-3], 'Open']))
           ): # current low lower than previous open
            
            cancel_orders(symbol)

            ticker = client.futures_symbol_ticker(symbol=symbol.upper())

            mark_price = round(float(ticker['price']), 3)

#             stop_price = df.loc[df.index[-1], 'Close']
#             if stop_price < mark_price:
#                 stop_price = mark_price
                
            take_profit_order = client.futures_create_order(
                symbol=symbol.upper(),
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                stopPrice=mark_price,
                stopLimitTimeInForce='GTC',
                closePosition = 'true',
                positionSide = 'LONG'
            )
            # winsound.Beep(frequency, duration)
            # print('Take profit created successfully')
            
        # else:
        #     print(f"Take profit target not reached bb_u:{round(df.loc[df.index[-1], 'bb_u'], 3)}")
            
    except Exception as e:
        print(f'Error creating TP order: {e}')

# from IPython.display import clear_output


# def console_log(df, symbol):
#     try:
#         df = df.reset_index(drop=True)
#         df = df.round(3)
#         print('----------------------------------------------------------------------------')
#         print(f"{symbol} - {str(df.loc[df.index[-1], 'Open_Time'])} at {str(df.loc[df.index[-1], 'Close'])}")
#         print()
#         print(df[['Open_Time', 'Open', 'Low', 'Close', 'Volume', 'c1', 'c2', 'c3', 'c4', 'bb_u', 'ema_18_4h', 'signal', 'entry']].tail(5))
#         print('----------------------------------------------------------------------------')
#         positions = client.futures_account()['positions']
#         for position in positions:
#             if float(position['positionAmt']) != 0:
#                 position_df = pd.DataFrame({'Symbol':position['symbol'],
#                                             'Side':position['positionSide'],
#                                             'Entry_P':round(float(position['entryPrice']),3),
#                                             'Amt':round(float(position['positionAmt']) * df.loc[df.index[-1], 'Close'],3),
#                                             'PL':round(float(position['unrealizedProfit']),3),
#                                             'X':round(float(position['leverage']),1),
#                                            }, index=[0])      
#                 print(position_df)


#     except Exception as e:
#         print(f'Error UPDATING info: {e}')    



current_bar_pos = False
def check_price(df, symbol):
    global current_k, current_bar_pos
    
    try:
        if ((df.iloc[df.index[-1]]['signal']) & # 三線條件成立
            (current_bar_pos == False)): # 本 K 未進場
            enter_position(df, symbol)
            # winsound.Beep(frequency, duration)
            df.loc[df.index[-1], 'entry'] = True
            current_bar_pos = True
            
        # 如果時間標籤改變，設 current_bar_pos 為可進場
        if df['Open_Time'][len(df) - 1] != current_k:

            print('time changed')
            current_k = df['Open_Time'][len(df) - 1]
            current_bar_pos = False
            
    except Exception as e:
        print(f'Error creating ENTRY order: {e}')


current_k = 0

def run(symbol):
    global current_k, stop_loss_p

    while True:

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

            # indicators(df, kd_dir)
            indicators(df)
            conditions(df)
            check_price(df, symbol)
            # df.to_csv(f'LONG_1h_ema_4h_ema_Multi.csv')
            # 更新狀態

            # 檢查停損
            try:
                positions_info = client.futures_account()['positions']
                long_positions = [p for p in positions_info if p['positionSide'] == 'LONG' and float(p['positionAmt']) != 0 and p['symbol'] == symbol.upper()]

                if long_positions:
                    check_tp(df, symbol)
                    check_sl(df, symbol, current_k)

                else:
                    stop_loss_p = 0
                    print(f'No LONG position. No SL for {symbol}')

            except Exception as e:
                print(f'Error checking SL for {symbol}: {e}')
          
        except ConnectionError as e:
            print("Connection error occurred:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)

        
        time.sleep(3.5)