
import websocket
import json
import pandas as pd
import ta
import matplotlib.pyplot as plt
import math

endpoint = 'wss://stream.binance.com:9443/ws'

symbol = 'ethusdt'
rate = 5

data = json.dumps({'method':'SUBSCRIBE','params':[symbol + '@kline_1m'],'id':1})

df = pd.DataFrame()

#indicators
def get_sma(close_p, rate = 30):
    return close_p.rolling(rate).mean()

def get_bollinger_bands(close_p, sma, rate = 30):
    std = close_p.rolling(rate).std()
    bband_up = sma + 2 * std
    bband_down = sma - 2 * std
    return bband_up, bband_down


def on_open(ws):
    ws.send(data)

def on_message(ws, message):
    global df, rate
    
    #get raw data
    out = json.loads(message)
    out = pd.DataFrame({'open':float(out['k']['o']),
                        'close':float(out['k']['c']),
                        'high':float(out['k']['h']),
                        'low':float(out['k']['l']),
                        'b_vol':float(out['k']['v']),
                        'closed':bool(out['k']['x']),
                        },
                        index=[pd.to_datetime(out['E'], unit = 'ms')])


    
    df = pd.concat([df,out], axis = 0)
    
    if df.shape[0] < rate:
        sma = df['close']
        bband_up, bband_down = df['close']
        df['sma'] = sma
        df['bb_u'] = bband_up
        df['bb_d'] = bband_down
    else:
        sma = get_sma(df['close'], rate)
        bband_up, bband_down = get_bollinger_bands(df['close'], sma, rate)
        df['sma'] = sma
        df['bb_u'] = bband_up
        df['bb_d'] = bband_down
    
    
    print(df)
    # df = df.tail(5)
    
    # df.drop(df[df['closed'] == False].index, axis=0, inplace=True)
    
    
    # plt.title(symbol + ' SMA_' + rate)
    # plt.xlabel('time')
    # plt.ylabel('Closing Prices')
    # plt.plot(df.close, label='Closing Prices')
    # plt.plot(sma, label= rate + ' SMA')
    # plt.plot(bband_up, label='Bollinger Up', c='g')
    # plt.plot(bband_down, label='Bollinger Down', c='r')
    # plt.legend()
    # plt.show()
    



ws = websocket.WebSocketApp(endpoint, on_message = on_message, on_open = on_open)
ws.run_forever()