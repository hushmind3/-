import numpy as np

def autocorrelation(series,max_lag=1000,lags=None):
    """
    From https://github.com/stakahashy/stylefact
    Autocorrelation function f(k)  measures the pearson correlation of two variables with lag k.
    
    Parameters
    _________
    series : array-like
        time-series to be evaluated
    max_lag : int, optional
        maximum lag evaluated

    Returns
    _______
    lags : list
        list of lags evaluated
    acf_values : list
        list of correlation values

    See also
    ________

    Examples
    ________

    References
    __________
    .. [Granger et al.] Granger, C.W. J., Ding, Z. Some Properties of Absolute Return: An Alternative Measure of Risk , Annales d'Économie et de Statistique, No. 40, pp. 67-91 1995
       [Cont] R. Cont, Empirical properties of asset returns: Stylized facts and statistical issues, Quantitative Finance, 1 (2001), pp. 1–14.

    """
    if lags is None:
        lags = [i+1 for i in range(max_lag)]
    acf_values = []
    mean = np.mean(series)
    var = np.var(series)
    for lag in lags:
        series_1 = series[:-lag]
        series_2 = series[lag:]
        value = np.mean((series_1-mean)*(series_2-mean))/var
        acf_values.append(value)
        
    return lags,acf_values


from scipy.special import gamma
from numpy.linalg import lstsq
from numpy import log2, exp2, pi, sqrt, cumsum, log, ones

def hurst(Z, adjust=True):
    """
    Hurst_RS(Z::AbstractVector) -> Tuple{Float64,Float64,Float64}

    Estimate the Hurst exponent via rescaled range analysis.

    # Arguments
    - `Z::AbstractVector`: the real-valued time series vector

    # Keywords
    - `adjust::Bool = true`: correct the estimate via the Anis-Lloyd adjustment

    # Returns
    - `Tuple{Float64,Float64,Float64}`: the Hurst exponent in the first position, the
        log(n) values in the second position, and the log(R/S) values in the third position.

    # Throws
    - `ArgumentError`: throws an error if the length of `Z` is less than 500.

    # Notes
    For several different lengths n, divide the time series `Z`` of length N into d subseries of length n, where n
    is an integral divisor of N. For each subseries m = 1, 2, ..., d, follow the steps presented in CITE.
    per confint paper -> log2, no n < 50, at least 512 observations

    # References
    https://watermark.silverchair.com/63-1-111.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAvUwggLxBgkqhkiG9w0BBwagggLiMIIC3gIBADCCAtcGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMZn9ZfD4HT8p48oH7AgEQgIICqFFc432q0RZTSB95JYv8I7i-nrSWUQdoHy2CfmhMYjbyoUXNCkoCV1Hu_MbJC28IYbaz0Li3n2PcmUeZjBkqoqec_OZLQMypFp5PCR2xxqRqOUW3VQP2lrvPH-pJe9znb-RBrgY2wNF0zkvxadG1f7GAntRh2JBchmRWoCLKANBerIZnzKlaufIpRMOAq6uajOo36gzoNvlBGI2WghtJhsZuQqCFwHYRcTtmefCGIJRr4xOXs7D1ap9F5OwL8NhsRuZ8pPq_0YQuQctiSjU2DRmyD0nJI1BybRb-678jD06sgI0XH7Hu_CWoUAVSqYHjuX1gvdYac1LHZ3nIq4AsNmVSPJzr3cZ9-C4rwJ-kmbZCjO2TXavD28xpLcwFuOzhjrwoQGLfhLFvYiyjJWLyUMnH4sFDS2QKR39IVX37qpN_6B3xipuowiAeY13dyzWw_zE3TjIGBumESD5UHOu3Q05yMi7DHlGZq-R9G2SxYJB4PnOOqgWvR7uMgCGw1Xjg0jWs9UrpTcDuQMTe0QoKjWUwjNTI5sodgI8_YBogBCCXankFtuqYSsiElfl5CI7CcWj33VX6H7sMv2zSE8j-XyuOZKGHA5VnvF4N2mjBupHbId7rh6olQ8ANub2jqKsv8colB2Zv6ygtVq0INA760Q3oduR2fl1TAc5Y-VxJ17GtWUjRPLkbgsBhITu6KpqRdQZIZnQIl7dUGh9TdLtxcuMkGEkLqSk3qsCQ02Vj7zhb5zwdk4L3dZ7UMU8Ov1YlUJExbkuRnDKJfTW_i9FOkDnYTbapjLbsWsbW4hJA_T1es65nEOfQVs2IaXu2Sbg6QLMiqHgPnVyauzMVFMhPtm-j2Coj0dc4zLFtRDct9aUxX5NNhrThVjwJurtXh9MhcKtWyAFMdq5u
    https://arxiv.org/pdf/1805.08931.pdf
    https://arxiv.org/pdf/cond-mat/0103510.pdf
    """
    # check series length
    N = len(Z)
    if N < 500:
        raise ValueError(f"Size Error: series length N = {N} < 500")
    
    # divide the data Z into various lengths n (for n > 50) and take log
    log_n = np.arange(log2(51), log2(N), 0.25)
    
    # collect each length n to evaluate
    n_list = [int(np.floor(np.exp2(x))) for x in log_n]
    
    # ensure the full data length N is included in evaluation set
    if N not in n_list:
        n_list.append(N)
        log_n = np.append(log_n, log2(N))
    
    # calculate expected value for R/S over various n
    RS_n = []
    for n in n_list:
        rs_m = []
        for start in range(0, N, n):
            if start + n > N:
                break
            # set the rolling subseries m within m = 1,2,...,d
            Z_m = Z[start:start+n]
            # calculate the mean of the subseries
            E_m = np.mean(Z_m)
            # normalize the data by subtracting the sample mean
            deviations = Z_m - E_m
            # sum the deviations from the mean
            Y_m = cumsum(deviations)
            # compute the range R
            R_m = max(Y_m) - min(Y_m)
            # compute the standard deviation S
            S_m = np.std(Z_m, ddof=1)
            # calculate the rescaled range R/S
            RS_part = R_m / S_m if R_m != 0 and S_m != 0 else 0
            # add RS of block
            if RS_part != 0:
                rs_m.append(RS_part)
        # store the mean RS value for all d subseries of length n
        if len(rs_m) > 0:
            RS_n.append(np.mean(rs_m))
    
    # use the relation (R/S)_n ~ cn^H to compute an estimate of H
    if adjust:
        # correct the biased estimate via the Anis-Lloyd adjustment
        E_RR = []
        for n in n_list:
            if n <= 340:
                AL_adjust = ((n - 0.5) / n) * (gamma((n - 1) / 2) / (sqrt(pi) * gamma(n / 2))) * \
                            sum(sqrt((n - i) / i) for i in range(1, n))
            else:
                AL_adjust = ((n - 0.5) / n) * (1 / (sqrt(n * (pi / 2)))) * \
                            sum(sqrt((n - i) / i) for i in range(1, n))
            E_RR.append(AL_adjust)
        # compute adjusted rescaled range statistic
        RS_AL = np.array(RS_n) - np.array(E_RR) + sqrt(0.5 * pi * np.array(n_list))
        A = np.vstack([log_n, np.ones(len(RS_AL))]).T
        RSlog = log2(RS_AL)
    else:
        A = np.vstack([log_n, np.ones(len(RS_n))]).T
        RSlog = log2(RS_n)
    
    B = RSlog
    # run simple linear regression -> slope is estimate of the hurst exponent
    H, c = lstsq(A, B, rcond=None)[0]
    c = exp2(c)
    
    return H, log_n, RSlog


def hurst_confint(N, level='ninetyfive'):
    """
    Hurst_confint(Z::AbstractVector; level::Symbol=:ninetyfive) -> Tuple{Float64,Float64}

    Returns the confidence interval for Hurst exponents estimated via rescaled range analysis.

    Returns a Tuple with the lower and upper bound of the Hurst exponent (according to the
    confidence interval).

    # Arguments
    - `Z::AbstractVector`: the real-valued time series vector

    # Keywords
    - `level::Symbol=:ninetyfive`: the confidence level (options -> `:ninety`,
        `:ninetyfive` (default), `:ninetynine`)

    # Returns
    - `Tuple{Float64,Float64,Float64}`: the lower bound of the Hurst exponent in the first
        position and the upper bound of the Hurst exponent in the second position (according to the
        confidence interval).

    # Throws
    - `ArgumentError`: throws an error if the length of `Z` is less than 500.

    # References
    https://arxiv.org/pdf/cond-mat/0103510.pdf
    """
    # set adjusted sample size -
    M = log2(N)

    if level == 'ninety':
        # 90% confidence interval
        H_lowerconfint = 0.5 - np.exp(-7.35 * log(log(M)) + 4.06)
        H_upperconfint = np.exp(-7.07 * log(log(M)) + 3.75) + 0.5
    elif level == 'ninetyfive':
        # 95% confidence interval
        H_lowerconfint = 0.5 - np.exp(-7.33 * log(log(M)) + 4.21)
        H_upperconfint = np.exp(-7.20 * log(log(M)) + 4.04) + 0.5
    elif level == 'ninetynine':
        # 99% confidence interval
        H_lowerconfint = 0.5 - np.exp(-7.19 * log(log(M)) + 4.34)
        H_upperconfint = np.exp(-7.51 * log(log(M)) + 4.58) + 0.5
    else:
        raise ValueError(f"Invalid Argument: confidence interval {level} not defined.")
    return H_lowerconfint, H_upperconfint