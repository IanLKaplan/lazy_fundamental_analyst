import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

plt.style.use('seaborn-whitegrid')


def df_concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrame objects with Datetime indexes. By forcing
    conversion to a Datetime index this can avoid mismatches on
    concatenation due to time (hours, minutes and seconds).
    :param df1:
    :param df2:
    :return:
    """
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)
    df_concat = pd.concat([df1, df2], axis=1)
    return df_concat


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
                back_start_date = self.start_date - timedelta(weeks=56)
                s_and_p_eps_raw = nasd.get("MULTPL/SP500_EARNINGS_YIELD_MONTH", start_date=back_start_date)
                eps_index = s_and_p_eps_raw.index
                ix_start = findDateIndex(eps_index, self.start_date - timedelta(weeks=52))
                assert ix_start >= 0
                s_and_p_eps = s_and_p_eps_raw[:][ix_start:]
                s_and_p_eps_yearly = s_and_p_eps.rolling(12).sum()
                ix_start = findDateIndex(eps_index, self.start_date)
                assert ix_start >= 0
                s_and_p_eps_yearly = round(s_and_p_eps_yearly[:][ix_start:], 2)
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

    start_year: the numerical year (e.g., 2007) as a string
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
bls_unemployment_df = round(bls_data.get_unemployment_data(), 2)


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
        ix = close_data.index
        ix = pd.to_datetime(ix)
        close_data.index = ix
        close_data.to_csv(file_path)
    assert len(close_data) > 0, f'Error reading data for {symbols}'
    return close_data


def increasing(data_df: pd.DataFrame, window: int) -> list:
    bullish_ix: List[int] = list()
    for i in range(window, data_df.shape[0]):
        if data_df.iloc[i].values[0] > data_df.iloc[i - window].values[0]:
            bullish_ix.append(i)
    return bullish_ix


def decreasing(data_df: pd.DataFrame, window: int) -> list:
    bearish_ix: List[int] = list()
    for i in range(window, data_df.shape[0]):
        if data_df.iloc[i].values[0] < data_df.iloc[i - window].values[0]:
            bearish_ix.append(i)
    return bearish_ix


def signal_dates(func: Callable, data_df: pd.DataFrame, window: int) -> DatetimeIndex:
    ix_l: List[int] = func(data_df, window)
    index = data_df.index
    dates = index[ix_l]
    return dates


def plot_hedge(instrument_df: pd.DataFrame, hedge_df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.plot(instrument_df, label=str(instrument_df.columns[0]))
    ax.plot(hedge_df, 'x', label=str(hedge_df.columns[0]))
    plt.title(title)
    ax.grid(visible=True)
    leg = ax.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')


def get_market_indexes(market_df: pd.DataFrame, index_dates: DatetimeIndex) -> List[int]:
    ix_l: List = list()
    market_index = market_df.index
    for date_ in index_dates:
        ix = findDateIndex(date_index=market_index, search_date=date_)
        if ix >= 0:
            ix_l.append(ix)
        else:
            print(f'Did not find date {date_}')
    return ix_l


spy_data_file = 'spy_adj_close.csv'
spy_close_df = get_market_data(file_name=spy_data_file,
                               data_col='Adj Close',
                               symbols=['spy'],
                               data_source='yahoo',
                               start_date=start_date,
                               end_date=end_date)

eps_bullish_dates = signal_dates(increasing, eps_yearly, window=3)
emp_bullish_dates = signal_dates(decreasing, bls_unemployment_df, window=3)

eps_bearish_dates: pd.DatetimeIndex = signal_dates(decreasing, eps_yearly, window=3)
emp_bearish_dates: pd.DatetimeIndex = signal_dates(increasing, bls_unemployment_df, window=3)

eps_ix_l = get_market_indexes(spy_close_df, eps_bearish_dates)
spy_eps_bear_df = spy_close_df.iloc[eps_ix_l]
# plot_hedge(spy_close_df, spy_eps_bear_df, 'EPS Momentum Bear Signal')
# plt.show()

emp_ix_l = get_market_indexes(spy_close_df, emp_bearish_dates)
spy_emp_bear_df = spy_close_df.iloc[emp_ix_l]
# plot_hedge(spy_close_df, spy_emp_bear_df, "Employment Momentum Bear Signal")
# plt.show()


if len(eps_bearish_dates) >= len(emp_bearish_dates):
    bearish_dates_ = emp_bearish_dates.isin(eps_bearish_dates)
    bearish_dates = emp_bearish_dates[bearish_dates_]
else:
    bearish_dates_ = eps_bearish_dates.isin(emp_bearish_dates)
    bearish_dates = eps_bearish_dates[bearish_dates_]

ix_l = get_market_indexes(spy_close_df, bearish_dates)
spy_bear_df: pd.DataFrame = spy_close_df.iloc[ix_l]
spy_bear_df.columns = ['Hedge']

# plot_hedge(spy_close_df, spy_bear_df, 'EPS and Unemployment Momentum Bear Signal')
# plt.show()

sh_data_file = 'sh_close.csv'
sh_close_df = get_market_data(file_name=sh_data_file,
                              data_col='Close',
                              symbols=['sh'],
                              data_source='yahoo',
                              start_date=start_date,
                              end_date=end_date)

qqq_data_file = 'qqq_close.csv'
qqq_close_df = get_market_data(file_name=qqq_data_file,
                               data_col='Close',
                               symbols=['qqq'],
                               data_source='yahoo',
                               start_date=start_date,
                               end_date=end_date)

spy_and_sh_df = df_concat(spy_close_df, sh_close_df)


# spy_and_sh_df.plot(grid=True, title='SPY and SH', figsize=(10,6))
# plt.show()

def find_month_periods(start_date: datetime, end_date: datetime, data: pd.DataFrame) -> pd.DataFrame:
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    date_index = data.index
    start_ix = findDateIndex(date_index, start_date)
    end_ix = findDateIndex(date_index, end_date)
    start_l = list()
    end_l = list()
    cur_month = start_date.month
    start_l.append(start_ix)
    i = 0
    for i in range(start_ix, end_ix + 1):
        date_i = convert_date(date_index[i])
        if date_i.month != cur_month:
            end_l.append(i - 1)
            start_l.append(i)
            cur_month = date_i.month
    end_l.append(i)
    # if there is not a full month period, remove the last period
    if end_l[-1] - start_l[-1] < 18:
        end_l.pop()
        start_l.pop()
    start_df = pd.DataFrame(start_l)
    end_df = pd.DataFrame(end_l)
    start_date_df = pd.DataFrame(date_index[start_l])
    end_date_df = pd.DataFrame(date_index[end_l])
    periods_df = pd.concat([start_df, start_date_df, end_df, end_date_df], axis=1)
    periods_df.columns = ['start_ix', 'start_date', 'end_ix', 'end_date']
    return periods_df


def collapse_asset_df(asset_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param asset_df: columns: asset, start_date, end_date
    :return:
    """
    row_l = list()
    row = asset_df[0:1]
    cur_asset = row['asset'][0]
    row_l.append(row)
    row_ix = 0
    for index in range(1, asset_df.shape[0]):
        next_row = asset_df[:][index:index + 1]
        next_asset = next_row['asset'][0]
        if next_asset == cur_asset:
            next_end_date = next_row['end_date'][0]
            last_row = row_l[row_ix]
            last_row.columns = asset_df.columns
            t_l = [last_row['asset'][0], last_row['start_date'][0], next_end_date]
            t_df = pd.DataFrame(t_l).transpose()
            t_df.columns = asset_df.columns
            row_l[row_ix] = t_df
        else:
            row_l.append(next_row)
            row_ix = row_ix + 1
            cur_asset = next_asset
    collapse_df = pd.DataFrame()
    for i in range(len(row_l)):
        collapse_df = pd.concat([collapse_df, row_l[i]], axis=0)
    return collapse_df


def get_asset_investments(risk_asset: pd.DataFrame,
                          bond_asset: pd.DataFrame,
                          spy_data: SpyData,
                          start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
    """
    :param risk_asset: the risk asset set
    :param bond_asset: the bond asset set
    :param spy_data:  SpyData object
    :param start_date: the start date for the period over which the calculation is performed.
    :param end_date: the end date for the period over which the calculation is performed.
    :return: a data frame with the columns: asset, start_date, end_date
            The asset will be the asset symbol (e.g., 'SPY', 'QQQ', etc)  The
            start date will be the start_date on which the asset should be purchased.
            The date is an ISO date in string format. The end_date is the date that
            the asset should be sold.
    """
    name_l: List = []
    date_l: List = []
    end_date_l: List = []
    month_periods = find_month_periods(start_date, end_date, risk_asset)
    for index, period in month_periods.iterrows():
        month_start_ix = period['start_ix']
        month_end_ix = period['end_ix']
        # back_start_ix is the start of the look back period used to calculate the highest returning asset
        back_start_ix = (month_start_ix - trading_quarter) if (month_start_ix - trading_quarter) >= 0 else 0
        period_start_date: datetime = convert_date(period['start_date'])
        period_end_date: datetime = convert_date(period['end_date'])
        date_l.append(period_start_date)
        end_date_l.append(period_end_date)
        asset_name = ''
        if spy_data.risk_state(period_start_date) == RiskState.RISK_ON:
            asset_name: str = chooseAssetName(back_start_ix, month_start_ix, risk_asset)
        else:  # RISK_OFF - bonds
            asset_name: str = chooseAssetName(back_start_ix, month_start_ix, bond_asset)
        name_l.append(asset_name)
    asset_df = pd.DataFrame([name_l, date_l, end_date_l]).transpose()
    asset_df.index = date_l
    asset_df.columns = ['asset', 'start_date', 'end_date']
    asset_df = collapse_asset_df(asset_df=asset_df)
    return asset_df


pass
