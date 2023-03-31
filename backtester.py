import itertools
import pandas as pd
import numpy as np

def backtest_all(start_time, interval_arr, kd_direction_arr, sl_atr_arr, tp_atr_arr, sl_determine_arr):
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