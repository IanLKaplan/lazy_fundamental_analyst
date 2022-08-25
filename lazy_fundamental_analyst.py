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

import requests
import json
import jsonpickle as jp


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


class BLSData:
    """
    A class that supports reading data from the Bureau of Labor Statistics (BLS)
    REST end point.

    This code is derived from the code published on the web page:
    https://www.bls.gov/developers/api_python.htm

    See also https://www.bd-econ.com/blsapi.html

    start_year: the numerical year (e.g., 2021) as a string
    end_year: same as start_year  start_year <= end_year
    """
    def __init__(self, start_year: str, end_year: str):
        self.start_year = start_year
        self.end_year = end_year
        self.unemployment_data_id = 'LNS14000000'
        self.bls_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
        self.headers = {'Content-type': 'application/json'}
        self.max_years = 10

    def http_request(self, start_year: int, end_year: int) -> str:
        request_json_str = {'seriesid': [self.unemployment_data_id],
                            'startyear': str(start_year),
                            'endyear': str(end_year)}
        request_json = json.dumps(request_json_str)
        http_data = requests.post(self.bls_url, data=request_json, headers=self.headers)
        return http_data.text

    def fetch_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        # The JSON for 'item' in the code below is:
        # {'year': '2016',
        # 'period': 'M12',
        # 'periodName': 'December',
        # 'value': '4.7',
        #  'footnotes': [{}]}
        #
        json_str = self.http_request(start_year, end_year)
        json_dict = jp.decode(json_str)
        status = json_dict['status']
        if status == 'REQUEST_NOT_PROCESSED':
            raise Exception(json_dict['message'])
        date_l = list()
        value_l = list()
        for series in json_dict['Results']['series']:
            for item in series['data']:
                year = item['year']
                period = item['period']
                value = float(item['value'])
                period_date = datetime(year=int(year), month=int(period[1:]), day=1)
                value_l.append(value)
                date_l.append(period_date)
        period_df = pd.DataFrame(value_l)
        period_df.index = date_l
        # Make sure that dates are in increasing order
        period_df.sort_index(inplace=True)
        return period_df

    def get_unemployment_data(self) -> pd.DataFrame:
        start_year_i = int(self.start_year)
        end_year_i = int(self.end_year)
        unemployment_df = pd.DataFrame()
        while start_year_i < end_year_i:
            period_end = min(((start_year_i + self.max_years) - 1), end_year_i)
            period_data_df = self.fetch_data(start_year_i, period_end)
            unemployment_df = pd.concat([unemployment_df, period_data_df], axis=0)
            delta = (period_end - start_year_i) + 1
            start_year_i = start_year_i + delta
        unemployment_df.columns = ['unemployment']
        return unemployment_df


bls_start_year: str = '2007'
bls_end_year: str = str(datetime.today().year)
bls_data = BLSData(bls_start_year, bls_end_year)
bls_unemployment_df = bls_data.get_unemployment_data()

bls_unemployment_df.plot(grid=True, title='Monthly Unemployment Rate (percent)', figsize=(10, 6))
plt.show()

pass
