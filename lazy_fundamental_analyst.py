from datetime import datetime, timedelta
from typing import List, Callable

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://data.nasdaq.com/tools/python
# pip install Nasdaq-Data-Link
import nasdaqdatalink as nasd
from pandas import DatetimeIndex


def convert_date(some_date):
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    return some_date


def findDateIndex(date_index: DatetimeIndex, search_date: datetime) -> int:
    '''
    In a DatetimeIndex, find the index of the date that is nearest to search_date.
    This date will either be equal to search_date or the next date that is less than
    search_date
    '''
    index: int = -1
    i = 0
    search_date = convert_date(search_date)
    date_t = datetime.today()
    for i in range(0, len(date_index)):
        date_t = convert_date(date_index[i])
        if date_t >= search_date:
            break
    if date_t > search_date:
        index = i - 1
    else:
        index = i
    return index


start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
# The "current date"
end_date: datetime = datetime.today() - timedelta(days=1)

# For all of the S&P ratios see https://data.nasdaq.com/data/MULTPL-sp-500-ratios
# https://data.nasdaq.com/data/MULTPL/SP500_EARNINGS_YIELD_MONTH-sp-500-earnings-yield-by-month
# Monthly EPS estimates from 1871(!) to present
s_and_p_eps_raw = nasd.get("MULTPL/SP500_EARNINGS_YIELD_MONTH")

eps_index = s_and_p_eps_raw.index
ix_start = findDateIndex(eps_index, start_date - timedelta(weeks=52))
s_and_p_eps = s_and_p_eps_raw[:][ix_start:]

# s_and_p_eps.plot(grid=True, title="Monthly S&P 500 Earnings per share", figsize=(10, 6))
# plt.show()

s_and_p_eps_yearly = s_and_p_eps.rolling(12).sum()
s_and_p_eps_yearly = s_and_p_eps_yearly[:][12:]

# s_and_p_eps_yearly.plot(grid=True, title="Yearly S&P 500 Earnings per share, by month", figsize=(10, 6))
# plt.show()


def bullish(data_df: pd.DataFrame, window: int) -> list:
    bullish_ix: List[int] = list()
    for i in range(window, data_df.shape[0]):
        if data_df.iloc[i].values[0] > data_df.iloc[i-window].values[0]:
            bullish_ix.append(i)
    return bullish_ix


def bearish(data_df: pd.DataFrame, window: int) -> list:
    bearish_ix: List[int] = list()
    for i in range(window, data_df.shape[0]):
        if data_df.iloc[i].values[0] < data_df.iloc[i-window].values[0]:
            bearish_ix.append(i)
    return bearish_ix


def signal_dates(func: Callable, data_df: pd.DataFrame, window: int) -> DatetimeIndex:
    ix = func(data_df, window)
    index = data_df.index
    dates = index[ix]
    return dates


bullish_dates = signal_dates(bullish, s_and_p_eps_yearly, window=3)
bearish_dates = signal_dates(bearish, s_and_p_eps_yearly, window=3)

pass
