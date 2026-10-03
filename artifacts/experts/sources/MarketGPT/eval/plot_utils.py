import pandas as pd
import numpy as np

NanosecondTime = int

def fmt_ts(timestamp: NanosecondTime) -> str:
    """
    Converts a timestamp stored as nanoseconds into a human readable string.
    """
    return pd.Timestamp(timestamp, unit="ns").strftime("%Y-%m-%d %H:%M:%S")


def str_to_ns(string: str) -> NanosecondTime:
    """
    Converts a human readable time-delta string into nanoseconds.

    Arguments:
        string: String to convert into nanoseconds. Uses Pandas to do this.

    Examples:
        - "1s" -> 1e9 ns
        - "1min" -> 6e10 ns
        - "00:00:30" -> 3e10 ns
    """
    return pd.to_timedelta(string).to_timedelta64().astype(int)


# https://github.com/stakahashy/stylefact/blob/master/stylefact/finance.py#L6
from bisect import bisect_left

def log_distribution(series,side='positive',ticks=None,sample_point=100):
    """
    Per ChatGPT:
        " The distribution calculated by this function is neither a CDF nor a CCDF. Instead, it computes
        a histogram-like distribution, where each dist_value represents the proportion of values in the
        series that fall within the corresponding interval defined by ticks. This distribution can be
        visualized as a probability mass function (PMF) if the data is discrete, or as a probability
        density function (PDF) if the data is continuous.

        What the Function Does
        It filters the series to only include positive or negative values based on the side parameter.
        It sorts the filtered series.
        It divides the sorted series into intervals (bins) defined by ticks.
        It calculates the proportion of data points within each bin.

        How to Plot the Distribution
        To plot the distribution computed by this function, you can create a bar plot or a line plot
        where the x-axis represents the tick intervals and the y-axis represents the proportion of
        data points within each interval. "


    Parameters
    _________
    series : array-like
       time-series to be evaluated
    side : str (positive,negative), optional
        the side to evaluate the tail
    ticks : array-like, optional
        hoge
    sample_point : int, optional
        If ticks is not specified, the number of ticks is set to this value
        Default 100

    Returns
    _______
    ticks : array-like
    dist_values : list
        the probability between ticks[i] and ticks[i+1]

    Examples
    ________

    References
    __________

    """
    assert side in ['positive','negative']
    if side == 'positive':
        series = series[series > 0]
        if ticks is None:
            ticks = np.linspace(0,np.max(series),num=sample_point)
    else:
        series = series[series < 0]
        if ticks is None:
            ticks = -np.linspace(0,-np.min(series),num=sample_point)[::-1]
    
    series = np.sort(series)
    dist_values = []
    
    for tick1,tick2 in zip(ticks[:-1],ticks[1:]):
        count = bisect_left(series,tick2)-bisect_left(series,tick1)
        value = count / series.size
        dist_values.append(value)
    return ticks,dist_values

# investigate alt impl -> https://github.com/jeffalstott/powerlaw/blob/master/powerlaw.py#L1888

# n = float(len(norm_returns))
# data = np.sort(norm_returns)
# data = data[data > 0]
# all_unique = not( any( data[:-1]==data[1:] ) )

# if all_unique:
#     CDF = np.arange(n)/n
# else:
# #This clever bit is a way of using searchsorted to rapidly calculate the
# #CDF of data with repeated values comes from Adam Ginsburg's plfit code,
# #specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
#     CDF = np.searchsorted(data, data,side='left')/n
#     unique_data, unique_indices = np.unique(data, return_index=True)
#     data=unique_data
#     CDF = CDF[unique_indices]

# survival = True
# if survival:
#     CDF = 1-CDF

# return data, CDF


# custom implentation
import numpy as np
from bisect import bisect_left
def calculate_cdf(series, side='positive', ticks=None, sample_point=100, survival=True):
    assert side in ['positive', 'negative']
    if side == 'positive':
        series = series[series > 0]
        if ticks is None:
            ticks = np.linspace(0, np.max(series), num=sample_point)
    else:
        series = series[series < 0]
        if ticks is None:
            # ticks = -np.linspace(0, -np.min(series), num=sample_point)[::-1]
            ticks = np.linspace(0, -np.min(series), num=sample_point)
        series = -series

    series = np.sort(series)
    dist_values = []

    for tick in ticks:
        count = bisect_left(series, tick)
        value = count / series.size
        dist_values.append(value)

    if survival:
        dist_values = 1 - np.array(dist_values)
        
    return ticks, dist_values
