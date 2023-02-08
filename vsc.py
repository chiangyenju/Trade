import websocket
import json
import pandas as pd


endpoint = 'wss://stream.binance.com:9443/ws'

data = json.dumps({'method':'SUBSCRIBE','params':['ethusdt@kline_1m'],'id':1})

df = pd.DataFrame()



def on_open(ws):
    ws.send(data)

def on_message(ws, message):
    global df, in_position
    out = json.loads(message)
    out = pd.DataFrame({'open':float(out['k']['o']),
                        'close':float(out['k']['c']),
                        'high':float(out['k']['h']),
                        'low':float(out['k']['l']),
                        'b_vol':float(out['k']['v']),
                        'q_vol':float(out['k']['q']),
                        'closed':bool(out['k']['x'])
                        },
                        index=[pd.to_datetime(out['E'], unit = 'ms')])
    df = pd.concat([df,out], axis = 0)
    df = df.tail(5)
    print(df)
    


ws = websocket.WebSocketApp(endpoint, on_message = on_message, on_open = on_open)
ws.run_forever()