
import websocket
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


timezone = 8
endpoint = 'wss://stream.binance.com:9443/ws'
symbol = 'ethusdt'
symbol_C = symbol.upper()
rate = 30
interval = '15m'
limit = rate*3

data = json.dumps({'method':'SUBSCRIBE','params':[symbol + '@kline_' + interval],'id':1})

df = pd.DataFrame()




def get_historical(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol_C,
              "interval": interval,
              "limit": limit}
    response = requests.get(url, params=params).json()
    df = pd.DataFrame(response, columns = ["Open time", "Open", "High", "Low", "Close",
                                           "Volume", "Close time", "Quote asset volume",
                                           "Number of trades", "Taker buy base asset volume",
                                           "Taker buy quote asset volume", "Ignore"])
    df = df[['Close time', 'Open', 'Close', "High", "Low", 'Volume']]
    convert_dict = {'Open': float, 'Close': float, "High": float, "Low": float, 'Volume': float}
    df = df.astype(convert_dict)
    return df

#indicators
def get_sma(close_p, rate = 30):
    return close_p.rolling(rate).mean()



def on_open(ws):
    ws.send(data)

def on_message(ws, message):
    global df, rate, limit
    out = json.loads(message)
    out = pd.DataFrame({'Open':float(out['k']['o']),
                        'Close':float(out['k']['c']),
                        'High':float(out['k']['h']),
                        'Low':float(out['k']['l']),
                        'Volume':float(out['k']['v']),
                        'Closed':bool(out['k']['x']),
                        },
                        index=[out['E']])
    print(out)

    # out['Sma'] = sma.tail(1)[0]
    # out['Bb_U'] = bband_up.tail(1)[0]
    # out['Bb_D'] = bband_down.tail(1)[0]
    df = pd.concat([df,out], axis = 0)
    
    
    print(df)
    # df = df.tail(5)
    # df.drop(df[df['closed'] == False].index, axis=0, inplace=True)


ws = websocket.WebSocketApp(endpoint, on_message = on_message, on_open = on_open)
ws.run_forever()