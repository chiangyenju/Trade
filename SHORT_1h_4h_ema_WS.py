import requests
import websocket
from IPython.display import clear_output

import json
import pandas as pd
import numpy as np
import datetime as dt
import time
import ta
import os


strategy_name = 'SHORT_1h_4h_ema_WS'

def send_telegram_message(token, chat_id, text):
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.get(api_url, params=params)
    return response.json()


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

client = Client(api_key, api_secret)

spot = client.get_account()
s_bal = pd.DataFrame(spot['balances'])
print(s_bal)


telegram_token = os.getenv('TELEGRAM_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHATID')
telegram_message = f'Start {strategy_name}'

send_telegram_message(telegram_token, telegram_chat_id, telegram_message)


timezone = -3

# step between timestamps in milliseconds, 60000 = 1min 
step = 60000 * 3600
start_time = round(time.time() * 1000 - (86400 * 1000 * 10))
end_time = round(time.time() * 1000)

def set_time():
    
    global start_time, end_time
    
    # start epoch till now, use prior 10 days for this strategy
    start_time = round(time.time() * 1000 - (86400 * 1000 * 10))
    end_time = round(time.time() * 1000)


interval_arr = ['1h', '4h']
ema_arr = [5, 22, 66]

ep_per = 1
sl_per = 1.02
sl_det = 'Close'


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

        dataframes[f'{symbol}_{interval}'] = raw_df


def get_klines(symbol, interval):
    global dataframes
    df = dataframes[f'{symbol}_{interval}']
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
            column_name = f'ema_{ema}_{interval}' # eth 1h 4th ema
            df[column_name] = ta.trend.EMAIndicator(raw_df.Close, window=ema, fillna=True).ema_indicator()

    # reset index and set current index as a column
    df = df.reset_index()

    # set new index with integers
    df = df.set_index(pd.RangeIndex(len(df)))

    # time_format(timezone)
    df['Open_Time'] = df['Open_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # fill up higher time frame empty values with equal interval between each value
    df = df.replace('', np.nan)
    df['ema_5_4h'] = df['ema_5_4h'].interpolate()
    df['ema_22_4h'] = df['ema_22_4h'].interpolate()
    df['ema_66_4h'] = df['ema_66_4h'].interpolate()
    symbol_dfs[f'df_{symbol}'] = df
    
    return symbol_dfs[f'df_{symbol}']


def indicators(df):    
    
# bb    
    bb_int = 30
    bb_dev = 2
    bb = ta.volatility.BollingerBands(df['Close'], window=bb_int, window_dev=bb_dev)
    df['bb_u'] = bb.bollinger_hband()
    df['bb_m'] = bb.bollinger_mavg()
    df['bb_l'] = bb.bollinger_lband()  


def conditions(df):
    
    df['c1'] = df['ema_5_1h'] <= df['ema_66_1h']
    df['c2'] = df['ema_66_1h'] <= df['ema_22_4h']
    df['c3'] = df['ema_5_1h'] <= df['ema_22_4h']
    df['c4'] = df['Close'] >= df['ema_22_4h'] * ep_per

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

    ema_22_val = df['ema_22_4h']
    
    df.loc[df.index[-1], 'entry_p'] = ema_22_val.loc[close_val.index[-1]] * ep_per
    df.loc[df.index[-1], 'stop_loss'] = ema_22_val.loc[close_val.index[-1]] * sl_per

    #-----position attributes-----#
    usdt_q = 300
    quantity = round(usdt_q / df.loc[df.index[-1], 'Close'], 0)

    if quantity <= 0:
        quantity = round(usdt_q / df.loc[df.index[-1], 'Close'], 2)
    
    cancel_orders(symbol)
    
    # 因為現貨會比期貨價格高，空單經常不會打回進場價，為確保進場，在此作調整
    ticker = client.futures_symbol_ticker(symbol=symbol.upper())

    mark_price = None
    
    if float(ticker['price']) >= 30:
        mark_price = round(float(ticker['price']), 2)
        stop_loss_p = round(df.loc[df.index[-1], 'stop_loss'], 2)

    elif float(ticker['price']) <= 2:
        mark_price = round(float(ticker['price']), 4)
        stop_loss_p = round(df.loc[df.index[-1], 'stop_loss'], 4)

    else:            
        mark_price = round(float(ticker['price']), 3)
        stop_loss_p = round(df.loc[df.index[-1], 'stop_loss'], 3)
        
    try:
        telegram_message = f'{symbol} E: {mark_price} SL: {stop_loss_p} via {strategy_name}'
        send_telegram_message(telegram_token, telegram_chat_id, telegram_message)
        
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
        print(f'Error enter_position: {e}')   
        time.sleep(1)

    return df


def check_sl(df, symbol, current_k):
    global stop_loss_p

    current_sl = stop_loss_p
    previous_high = df.loc[df.index[-2], 'High']

    print(f'{symbol} SL at {current_sl}, previous High {previous_high}')
    

    # 確定停損
    if (previous_high >= current_sl) and (current_sl != 0):
                
        cancel_orders(symbol)

        # 如果目前價格高於停損
        ticker = client.futures_symbol_ticker(symbol=symbol.upper())

        mark_price = None
        
        if float(ticker['price']) >= 30:
            mark_price = round(float(ticker['price']), 2)
    
        elif float(ticker['price']) <= 2:
            mark_price = round(float(ticker['price']), 4)

        else:            
            mark_price = round(float(ticker['price']), 3)   
        
        try:
            telegram_message = f'ALERT SL - {symbol}: {stop_loss_p} via {strategy_name}'
            send_telegram_message(telegram_token, telegram_chat_id, telegram_message)
            stop_loss_p = 0
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
            print(f'Error check_sl: {e}')   
            time.sleep(1)


def check_tp(df, symbol):
    
    try:
        # 隨時偵測出場條件是否成立，不必等 Close 發生
        if (
            (df.loc[df.index[-1], 'Close'] <= df.loc[df.index[-1], 'bb_l']) or 
            (
                (df.loc[df.index[-2], 'Volume'] > df.loc[df.index[-3], 'Volume']) and
                (df.loc[df.index[-3], 'Open'] > df.loc[df.index[-3], 'Close']) and
                (df.loc[df.index[-2], 'Open'] < df.loc[df.index[-2], 'Close']) and
                (df.loc[df.index[-2], 'High'] > df.loc[df.index[-3], 'Open'])
            )
        ):
            
            cancel_orders(symbol)

            ticker = client.futures_symbol_ticker(symbol=symbol.upper())

            mark_price = None
            
            if float(ticker['price']) >= 30:
                mark_price = round(float(ticker['price']), 2)
        
            elif float(ticker['price']) <= 2:
                mark_price = round(float(ticker['price']), 4)
    
            else:            
                mark_price = round(float(ticker['price']), 3)
            
            try: 
                take_profit_order = client.futures_create_order(
                    symbol=symbol.upper(),
                    side='BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=mark_price,
                    stopLimitTimeInForce='GTC',
                    closePosition = 'true',
                    positionSide = 'SHORT'
                )
            
            except:
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
        print(f'Error check_tp: {e}')   
        time.sleep(2)


current_bar_pos = False
def check_price(df, symbol):
    global current_k, current_bar_pos
    
    try:
        if ((df.iloc[df.index[-1]]['signal']) and (current_bar_pos == False)): # 本 K 未進場
            enter_position(df, symbol)
            df.loc[df.index[-1], 'entry'] = True
            current_bar_pos = True
            
        # 如果時間標籤改變，設 current_bar_pos 為可進場
        if df['Open_Time'][len(df) - 1] != current_k:

            current_k = df['Open_Time'][len(df) - 1]
            # current_bar_pos = False
            
    except Exception as e:
        print(f'Error check_price: {e}')   
        time.sleep(1)


current_k = 0
line_count = 0
max_lines = 2
symbol = ''

def update_dataframe(df, open_time, open_price, close_price, high_price, low_price, volume):
    global line_count, max_lines
    if df.loc[df.index[-1], 'Open_Time'] == open_time:
        try: 
            df.loc[df.index[-1]] = [open_time, open_price, close_price, high_price, low_price, volume]
            # print(df.tail(2))
            
            line_count += 1
            if line_count >= max_lines:
                clear_output(wait=True)
                line_count = 0
                
        except Exception as e:
            print(f'error update_dataframe try {e}')
    else:
        try: 
            create_raw(symbol, interval_arr, step)
        except Exception as e:
            print(f'error update_dataframe else {e}')


reconnect = True
retry_count = 0
max_retry = 12
intv1_update = False
intv2_update = False


def on_message_wrapper(symbol):
    
    create_raw(symbol, interval_arr, step)

    def on_message(ws, message):
        global current_k, line_count, max_lines, stop_loss_p, intv1_update, intv2_update, current_bar_pos
        # Handle incoming messages
        data = json.loads(message)
        try:
    
            set_time()
            
            kline_data = data['k']
            # symbol = kline_data['s']
            interval = kline_data['i']
            open_time = kline_data['t']
            open_price = kline_data['o']
            close_price = kline_data['c']
            high_price = kline_data['h']
            low_price = kline_data['l']
            volume = kline_data['v']
    
            if interval == interval_arr[0] and intv1_update == False:
                update_dataframe(dataframes[f'{symbol}_{interval}'], open_time, open_price, close_price, high_price, low_price, volume)
                intv1_update = True
            if interval == interval_arr[1] and intv2_update == False:
                update_dataframe(dataframes[f'{symbol}_{interval}'], open_time, open_price, close_price, high_price, low_price, volume)
                intv2_update = True
                
            if intv1_update and intv2_update:
                
                df = multi_timeframes(symbol)
                # df = dataframes[f'{symbol}_{intv}']
        
                indicators(df)
                conditions(df)
                check_price(df, symbol)
                
                try:
                    positions_info = client.futures_account()['positions']
                    no_short_positions = [p for p in positions_info if p['positionSide'] == 'SHORT' and float(p['positionAmt']) == 0 and p['symbol'] == symbol.upper()]
        
                    if no_short_positions:
                    #     check_tp(df, symbol)
                        # check_sl(df, symbol, current_k)
                        current_bar_pos = False
                        
                    # else:
                    #     stop_loss_p = 0
                    ot = df.loc[df.index[-1], 'Open_Time']
                    cp = df.loc[df.index[-1], 'Close']
 
                    print(f'{symbol}, {ot}, {cp}')
        
                    intv1_update = False
                    intv2_update = False
                        
                except Exception as e:
                    print(f'Error WS for {symbol}: {e}')
              
        except ConnectionError as e:
            print("Connection error occurred:", e)
            time.sleep(2)

    return on_message
        
def on_error(ws, error):
    global reconnect, retry_count
    print(f"WebSocket Error: {error}")
    reconnect = True
    retry_count += 1
    if retry_count >= max_retry:
        print("Max retry attempts reached. Stopping reconnection.")
        reconnect = False
        ws.close()
    else:
        print(f"Reconnecting... Retry #{retry_count}")

def on_close(ws):
    global reconnect
    print("WebSocket connection closed")
    if reconnect:
        print("Reconnecting...")
        ws.run_forever()

def on_open(ws):
    global retry_count
    # Reset retry count
    retry_count = 0
    # Subscribe to the one-hour Kline stream
    ws.send(f'{{"method": "SUBSCRIBE", "params": ["{symbol}@kline_1h"], "id": 1}}')
    # Subscribe to the four-hour Kline stream
    ws.send(f'{{"method": "SUBSCRIBE", "params": ["{symbol}@kline_4h"], "id": 2}}')

def run(symbol_run):
    
    global symbol
    
    # WebSocket connection URL
    url = "wss://stream.binance.com:9443/ws"
    symbol = symbol_run
    
    while reconnect:
        # Create a WebSocket connection
        websocket.enableTrace(False)  # Uncomment to enable tracing/debugging
        ws = websocket.WebSocketApp(url, on_error=on_error, on_close=on_close)
        ws.on_message = on_message_wrapper(symbol)
        ws.on_open = on_open
    
        # Start the WebSocket connection
        ws.run_forever()