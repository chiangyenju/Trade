#%%
import multiprocessing
import itertools
import time
import pandas as pd
import backtest
from backtest import start_time_arr, interval_arr, kd_direction_arr, sl_atr_arr, tp_atr_arr, sl_determine_arr

#%%
if __name__ ==  '__main__': 
    pool = multiprocessing.Pool(8)
    loop_start_time = time.time()
    results_list = pool.map(backtest.run_backtest, [(start_time, interval, kd_direction, sl_atr, tp_atr, sl_determine) for start_time in start_time_arr for interval in interval_arr for kd_direction in kd_direction_arr for sl_atr in sl_atr_arr for tp_atr in tp_atr_arr for sl_determine in sl_determine_arr])
    loop_end_time = time.time()
    print("Time taken to execute for loop:", loop_end_time - loop_start_time, "seconds")
    pool.close()
    pool.join()
    
    results_df = pd.concat(results_list, ignore_index=True)
    print(results_df)