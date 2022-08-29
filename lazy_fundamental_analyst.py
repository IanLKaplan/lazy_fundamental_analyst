import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://data.nasdaq.com/tools/python
# pip install Nasdaq-Data-Link
import nasdaqdatalink as nasd
from pandas import DatetimeIndex

# pip install pandas-datareader
from pandas_datareader import data

import requests
import json
import jsonpickle as jp

import tempfile


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


class NASDQData:
    """
    eps_start_date: the data that the rolling 12-month sum should start at
    """
    def __init__(self, eps_start_date: datetime):
        self.start_date = eps_start_date
        self.S_AND_P_EARNINGS_KEY = "MULTPL/SP500_EARNINGS_YIELD_MONTH"
        self.DATALINK_KEY = 'NASDAQ_DATA_LINK_API_KEY'
        self.s_and_p_earnings_file = "s_and_p_earnings.csv"

    def get_s_and_p_earnings(self) -> pd.DataFrame:
        temp_root: str = tempfile.gettempdir() + '/'
        file_path: str = temp_root + self.s_and_p_earnings_file
        temp_file_path = Path(file_path)
        file_size = 0
        if temp_file_path.exists():
            file_size = temp_file_path.stat().st_size

        if file_size > 0:
            s_and_p_eps_yearly = pd.read_csv(file_path, index_col='Date')
        else:
            nasdaq_datalink_key = os.environ.get(self.DATALINK_KEY)
            if nasdaq_datalink_key == None:
                print("Warning: no NASDAQ data link key has been set in the environment")
            # For all of the S&P ratios see https://data.nasdaq.com/data/MULTPL-sp-500-ratios
            # https://data.nasdaq.com/data/MULTPL/SP500_EARNINGS_YIELD_MONTH-sp-500-earnings-yield-by-month
            try:
                back_start_date =  self.start_date - timedelta(weeks=56)
                s_and_p_eps_raw = nasd.get("MULTPL/SP500_EARNINGS_YIELD_MONTH", start_date=back_start_date)
                eps_index = s_and_p_eps_raw.index
                ix_start = findDateIndex(eps_index, self.start_date - timedelta(weeks=52))
                assert ix_start >= 0
                s_and_p_eps = s_and_p_eps_raw[:][ix_start:]
                s_and_p_eps_yearly = s_and_p_eps.rolling(12).sum()
                s_and_p_eps_yearly = round(s_and_p_eps_yearly[:][12:], 2)
                s_and_p_eps_yearly.to_csv(file_path)
            except Exception as e:
                raise Exception(f'nasdaq-data-link error: {str(e)}')

        return s_and_p_eps_yearly


# The data returned will be the rolling yearly sum, so the start_date is backed up by a year to properly start
# on start-date.
nasdq_data = NASDQData(start_date)
eps_yearly = nasdq_data.get_s_and_p_earnings()
# eps_yearly.plot(grid=True, title="Yearly S&P 500 Earnings per share, by month", figsize=(10, 6))
# plt.show()


class BLSData:
    """
    A class that supports reading data from the Bureau of Labor Statistics (BLS)
    REST end point.

    This code is derived from the code published on the web page:
    https://www.bls.gov/developers/api_python.htm

    See also https://www.bd-econ.com/blsapi.html

    start_year: the numerical year (e.g., 2021) as a string
    end_year: same as start_year  start_year <= end_year

    This class writes the data out to a temp file, so that the file can be read
    in subsequent runs.  This avoids running into the BLS daily download limit.
    This also improves performance.
    """
    def __init__(self, start_year: str, end_year: str):
        self.start_year = start_year
        self.end_year = end_year
        self.unemployment_data_id = 'LNS14000000'
        self.bls_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
        self.headers = {'Content-type': 'application/json'}
        self.max_years = 10
        self.bls_file_name = 'bls_monthly_unemployment.csv'

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
        if status != 'REQUEST_SUCCEEDED':
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

    def get_unemployment_data_from_bls(self) -> pd.DataFrame:
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
        unemployment_df.index.name = 'Date'
        return unemployment_df

    def get_unemployment_data(self) -> pd.DataFrame:
        temp_root: str = tempfile.gettempdir() + '/'
        file_path: str = temp_root + self.bls_file_name
        temp_file_path = Path(file_path)
        file_size = 0
        if temp_file_path.exists():
            file_size = temp_file_path.stat().st_size

        if file_size > 0:
            unemployment_data_df = pd.read_csv(file_path, index_col='Date')
        else:
            unemployment_data_df = self.get_unemployment_data_from_bls()
            unemployment_data_df.to_csv(file_path)
        return unemployment_data_df


bls_start_year: str = '2007'
bls_end_year: str = str(datetime.today().year)
bls_data = BLSData(bls_start_year, bls_end_year)
#
# Round to a whole number since fractional unemployment values are not very accurate
# (e.g., there is a lot of noise in unemployment numbers)
bls_unemployment_df = round(bls_data.get_unemployment_data(), 0)

# bls_unemployment_df.plot(grid=True, title='Monthly Unemployment Rate (percent)', figsize=(10, 6))
# plt.show()

def get_market_data(file_name: str,
                    data_col: str,
                    symbols: List,
                    data_source: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.DataFrame:
    """
      file_name: the file name in the temp directory that will be used to store the data
      data_col: the type of data - 'Adj Close', 'Close', 'High', 'Low', 'Open', Volume'
      symbols: a list of symbols to fetch data for
      data_source: yahoo, etc...
      start_date: the start date for the time series
      end_date: the end data for the time series
      Returns: a Pandas DataFrame containing the data.

      If a file of market data does not already exist in the temporary directory, fetch it from the
      data_source.
    """
    temp_root: str = tempfile.gettempdir() + '/'
    file_path: str = temp_root + file_name
    temp_file_path = Path(file_path)
    file_size = 0
    if temp_file_path.exists():
        file_size = temp_file_path.stat().st_size

    if file_size > 0:
        close_data = pd.read_csv(file_path, index_col='Date')
    else:
        if type(symbols) == str:
            t = list()
            t.append(symbols)
            symbols = t
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.DataFrame = panel_data[data_col]
        close_data.to_csv(file_path)
    assert len(close_data) > 0, f'Error reading data for {symbols}'
    return close_data


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


spy_data_file = 'spy_adj_close.csv'
spy_close_df = get_market_data(file_name=spy_data_file,
                               data_col='Adj Close',
                               symbols=['spy'],
                               data_source='yahoo',
                               start_date=start_date,
                               end_date=end_date)


eps_bullish_dates = signal_dates(bullish, eps_yearly, window=3)
emp_bullish_dates = signal_dates(bullish, bls_unemployment_df, window=3)

eps_bearish_dates = signal_dates(bearish, eps_yearly, window=3)
emp_bearish_dates = signal_dates(bearish, bls_unemployment_df, window=3)

bearish_dates_1 = eps_bearish_dates.isin(emp_bearish_dates)
bearish_dates_2 = emp_bearish_dates.isin(eps_bearish_dates)

pass
