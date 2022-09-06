import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Callable, Tuple

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

trading_days = 252


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


# spy_and_sh_df = df_concat(spy_close_df, sh_close_df)


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


def hedge_portfolio(yearly_eps: pd.DataFrame, unemployment: pd.DataFrame, start_date: datetime) -> bool:
    """
    All dates for the EPS data and the unemployment date are on the first of the month
    :param start_date:
    :return:
    """
    window = 3
    index_date = datetime(start_date.year, start_date.month, 1)
    eps_index = yearly_eps.index
    eps_ix_now = findDateIndex(eps_index, index_date)
    eps_ix_past = eps_ix_now - window
    assert eps_ix_past >= 0
    unemployment_index = unemployment.index
    emp_ix_now = findDateIndex(unemployment_index, index_date)
    emp_ix_past = emp_ix_now - window
    assert emp_ix_past >= 0
    hedge = False
    # If yearly S&P 500 earnings per share is lower than three months ago and
    # unemployment is higher than it was three months ago, then hedge is true, otherwise, false
    if (yearly_eps.iloc[eps_ix_now].values[0] < yearly_eps.iloc[eps_ix_past].values[0]) and \
            (unemployment.iloc[emp_ix_now].values[0] > unemployment.iloc[emp_ix_past].values[0]):
        hedge = True
    return hedge


def get_portfolio_hedge(portfolio_asset: pd.DataFrame,
                        hedge_asset_sym: str,
                        start_date: datetime,
                        end_date: datetime,
                        yearly_eps: pd.DataFrame,
                        unemployment: pd.DataFrame) -> pd.DataFrame:
    """
    :param portfolio_asset:
    :param hedge_asset_sym:
    :param start_date:
    :param end_date:
    :return:
    """
    name_l: List = []
    date_l: List = []
    end_date_l: List = []
    month_periods = find_month_periods(start_date, end_date, portfolio_asset)
    for index, period in month_periods.iterrows():
        period_start_date: datetime = convert_date(period['start_date'])
        period_end_date: datetime = convert_date(period['end_date'])
        date_l.append(period_start_date)
        end_date_l.append(period_end_date)
        asset_name = portfolio_asset.columns[0]
        if hedge_portfolio(yearly_eps, unemployment, period_start_date):
            asset_name = hedge_asset_sym
        name_l.append(asset_name)
    asset_df = pd.DataFrame([name_l, date_l, end_date_l]).transpose()
    asset_df.index = date_l
    asset_df.columns = ['asset', 'start_date', 'end_date']
    asset_df = collapse_asset_df(asset_df=asset_df)
    return asset_df


def simple_return(time_series: np.array, period: int = 1) -> List:
    return list(((time_series[i] / time_series[i - period]) - 1.0 for i in range(period, len(time_series), period)))


def return_df(time_series_df: pd.DataFrame) -> pd.DataFrame:
    r_df: pd.DataFrame = pd.DataFrame()
    time_series_a: np.array = time_series_df.values
    return_l = simple_return(time_series_a, 1)
    r_df = pd.DataFrame(return_l)
    date_index = time_series_df.index
    r_df.index = date_index[1:len(date_index)]
    r_df.columns = time_series_df.columns
    return r_df


def adjust_time_series(ts_one_df: pd.DataFrame, ts_two_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjust two DataFrame time series with overlapping date indices so that they
    are the same length with the same date indices.
    """
    ts_one_index = pd.to_datetime(ts_one_df.index)
    ts_two_index = pd.to_datetime(ts_two_df.index)
    # filter the close prices
    matching_dates = ts_one_index.isin(ts_two_index)
    ts_one_adj = ts_one_df[matching_dates]
    # filter the rf_prices
    ts_one_index = pd.to_datetime(ts_one_adj.index)
    matching_dates = ts_two_index.isin(ts_one_index)
    ts_two_adj = ts_two_df[matching_dates]
    return ts_one_adj, ts_two_adj


def apply_return(start_val: float, return_df: pd.DataFrame) -> np.array:
    port_a: np.array = np.zeros(return_df.shape[0] + 1)
    port_a[0] = start_val
    return_a = return_df.values
    for i in range(1, len(port_a)):
        port_a[i] = port_a[i - 1] + port_a[i - 1] * return_a[i - 1]
    return port_a


def build_plot_data(holdings: float, portfolio_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    t_port_df, t_spy_df = adjust_time_series(portfolio_df, spy_df)
    spy_return = return_df(t_spy_df)
    spy_return_a = apply_return(start_val=holdings, return_df=spy_return)
    spy_port = pd.DataFrame(spy_return_a)
    spy_port.columns = ['SPY']
    spy_port.index = pd.to_datetime(t_spy_df.index)
    plot_df = t_port_df.copy()
    plot_df['SPY'] = spy_port
    return plot_df


def hedge_return(portfolio_asset: pd.DataFrame,
                 hedge_asset: pd.DataFrame,
                 start_date: datetime,
                 end_date: datetime,
                 yearly_eps: pd.DataFrame,
                 unemployment: pd.DataFrame) -> pd.DataFrame:
    portfolio_start = datetime(start_date.year, start_date.month, start_date.day)

    # asset_periods has the columns ['asset', 'start_date', 'end_date']
    asset_periods = get_portfolio_hedge(portfolio_asset=portfolio_asset, hedge_asset_sym=hedge_asset.columns[0],
                                        start_date=portfolio_start, end_date=end_date, yearly_eps=yearly_eps,
                                        unemployment=unemployment)
    composit_df = pd.concat([portfolio_asset, hedge_asset], axis=1)
    portfolio_index = composit_df.index
    all_return = pd.DataFrame()
    for index, period in asset_periods.iterrows():
        period_start_date = period['start_date']
        period_end_date = period['end_date']
        period_start_ix = findDateIndex(date_index=portfolio_index, search_date=period_start_date)
        period_end_ix = findDateIndex(date_index=portfolio_index, search_date=period_end_date)
        asset = period['asset']
        period_s = composit_df[asset][period_start_ix - 1:period_end_ix + 1]
        period_df = pd.DataFrame(period_s)
        period_df.columns = ['return']
        period_return_df = return_df(period_df)
        all_return = pd.concat([all_return, period_return_df], axis=0)
    start_period = asset_periods.iloc[0]
    end_period = asset_periods.iloc[-1]
    range_start_date = start_period['start_date']
    range_end_date = end_period['end_date']
    range_start_ix = findDateIndex(date_index=portfolio_index, search_date=range_start_date)
    range_end_ix = findDateIndex(date_index=portfolio_index, search_date=range_end_date)
    result_index = portfolio_index[range_start_ix:range_end_ix + 1]
    all_return.index = result_index
    return all_return


def asset_return(asset_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    asset_index = asset_df.index
    start_ix = findDateIndex(date_index=asset_index, search_date=start_date)
    end_ix = findDateIndex(date_index=asset_index, search_date=end_date)
    assert start_ix > 0 and end_ix > 0
    period_df = asset_df[:][start_ix - 1:end_ix + 1]
    period_r = return_df(period_df)
    return period_r


def get_start_end_dates(data_df: pd.DataFrame) -> Tuple[datetime, datetime]:
    num_rows = data_df.shape[0]
    first_row = data_df[:][0:1]
    last_row = data_df[:][num_rows - 1:num_rows]
    start_date = convert_date(first_row.index[0])
    end_date = convert_date(last_row.index[0])
    return start_date, end_date


test_start_date_str = '2008-01-02'
test_start_date: datetime = datetime.fromisoformat(test_start_date_str)

def hedged_portfolio(holdings: int,
                     portfolio_asset: pd.DataFrame,
                     hedge_asset: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime,
                     yearly_eps: pd.DataFrame,
                     unemployment: pd.DataFrame) -> pd.DataFrame:

    hedge_r = hedge_return(portfolio_asset=portfolio_asset,
                           hedge_asset=hedge_asset,
                           start_date=start_date,
                           end_date=end_date,
                           yearly_eps=yearly_eps,
                           unemployment=unemployment)

    hedge_start_date, hedge_end_date = get_start_end_dates(hedge_r)

    asset_r = asset_return(asset_df=portfolio_asset, start_date=hedge_start_date, end_date=hedge_end_date)

    total_r_a = (hedge_r.values + asset_r.values) / 2.0
    total_r_df = pd.DataFrame(total_r_a)
    total_r_df.index = hedge_r.index
    column_str = f'{portfolio_asset.columns[0]}/{hedge_asset.columns[0]}'
    total_r_df.columns = [column_str]
    port_a = apply_return(start_val=holdings, return_df=total_r_df[:][1:])
    port_df = pd.DataFrame(port_a)
    port_df.index = total_r_df.index
    port_df.columns = total_r_df.columns
    return port_df


holdings = 100000
port_start_date_str = '2008-01-03'
port_start_date: datetime = datetime.fromisoformat(port_start_date_str)
port_df = hedged_portfolio(holdings=holdings,
                           portfolio_asset=qqq_close_df,
                           hedge_asset=sh_close_df,
                           start_date=port_start_date,
                           end_date=end_date,
                           yearly_eps=eps_yearly,
                           unemployment=bls_unemployment_df)
plot_df = build_plot_data(holdings=holdings, portfolio_df=port_df, spy_df=spy_close_df)
plot_df.plot(grid=True, title='QQQ/SH and SPY', figsize=(10, 6))
plt.show()

pass
