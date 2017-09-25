# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""

import numpy as np


# Without the means
def squared_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
       (array-like) Squared Error
    """
    y_true, y_estimated = map(np.asarray, [y_true, y_estimated])
    return (y_true - y_estimated) ** 2


def absolute_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (array-like) Absolute Error
    """
    y_true, y_estimated = map(np.asarray, [y_true, y_estimated])
    return np.abs(y_true - y_estimated)


def absolute_percentage_error(y_true, y_estimated):
    """
    Known issues:
        - It is not symmetric since:
            - y_true, y_estimated do not play a symmetric role
            - over- and under-forecasts are not treated equally
        - If the actual value or forecast value is 0, the value of error will boom up to the upper-limit of error

    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
       (array-like) Absolute Percentage Error
    """
    y_true, y_estimated = map(np.asarray, [y_true, y_estimated])
    return np.abs((y_true - y_estimated) / y_true)


def adjusted_absolute_percentage_error(y_true, y_estimated):
    """
    Known issues:
        - It is not symmetric since over- and under-forecasts are not treated equally
        - If the actual value or forecast value is 0, the value of error will boom up to the upper-limit of error

    Args:
        y_true (array-like):
        y_estimated (array-like):

    References:
         https://www.wikiwand.com/en/Symmetric_mean_absolute_percentage_error

    Returns:
        (array-like) Adjusted Absolute Percentage Error
    """
    y_true, y_estimated = map(np.asarray, [y_true, y_estimated])
    return np.abs((y_true - y_estimated) / (np.abs(y_true) + np.abs(y_estimated)))


def absolute_scaled_error(y_train, y_true, y_estimated, seasonal_period=1):
    """
    Properties:
        - Scale invariance (since error fraction)
        - Predictable behavior towards 0
        - Symmetry (both symmetry criterion verified)

    Args:
        seasonal_period (int): default equal to 1 (non-seasonal time series)
        y_train (array-like):
        y_true (array-like):
        y_estimated (array-like):

    References:
        https://www.wikiwand.com/en/Mean_absolute_scaled_error
        "Another look at measures of forecast accuracy", Rob J Hyndman

    Returns:
        (array-like) Absolute Scaled Error
    """
    y_true, y_estimated, y_train = map(np.asarray, [y_true, y_estimated, y_train])
    # Mean Absolute Error of the naïve forecast method on the training set
    # which uses the actual value from the prior period as the forecast: Forecasted[t] = Actual[t−1]
    mae_naive_forecast = np.mean(absolute_error(y_true=y_train[seasonal_period:],
                                                y_estimated=y_train[:-seasonal_period]))

    # Absolute error of the forecast
    ae_forecast = absolute_error(y_true=y_true, y_estimated=y_estimated)
    return ae_forecast / mae_naive_forecast


# Mean errors
def mean_squared_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (float) MSE
    """
    return np.mean(squared_error(y_true=y_true, y_estimated=y_estimated))


def mean_absolute_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (float) MAE
    """
    return np.mean(absolute_error(y_true=y_true, y_estimated=y_estimated))


def mean_absolute_percentage_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (float) MAPE
    """
    return np.mean(absolute_percentage_error(y_true=y_true, y_estimated=y_estimated))


def mean_adjusted_absolute_percentage_error(y_true, y_estimated):
    """
    Args:
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (float) MAAPE
    """
    return np.mean(adjusted_absolute_percentage_error(y_true=y_true, y_estimated=y_estimated))


def mean_absolute_scaled_error(y_train, y_true, y_estimated, seasonal_period=1):
    """
    Args:
        seasonal_period (int): default equal to 1 (non-seasonal time series)
        y_true (array-like):
        y_estimated (array-like):

    Returns:
        (float) MASE
    """
    return np.mean(
        absolute_scaled_error(y_train=y_train, y_true=y_true, y_estimated=y_estimated, seasonal_period=seasonal_period))


# Decorator to have metric ignore NaNs
def nan_ignoring(metric):
    """
    Decorator to make metric ignore NaNs

    Args:
        metric (callable): metric to modify

    Returns:
        nan-ignoring metric function
    """

    def nan_wrapper(y_true, y_estimated):
        """
        Wrapping function

        Args:
            y_true (array-like):
            y_estimated (array-like):

        Returns:
            nan-ignoring metric function value for (y_true, y_estimated)
        """
        # Boolean mask
        # True is y_true or y_estimated are NaN
        # False otherwise
        nan_mask = np.logical_or(np.isnan(y_true), np.isnan(y_estimated))
        # New arguments to the metric function: values without NaNs
        kwargs = {
            'y_true': y_true[~nan_mask],
            'y_estimated': y_estimated[~nan_mask]
        }
        return metric(**kwargs)

    return nan_wrapper


# Means ignoring NaN
@nan_ignoring
def mean_squared_error_ignoring_nans(y_true, y_estimated):
    return mean_squared_error(y_true=y_true, y_estimated=y_estimated)


@nan_ignoring
def mean_absolute_error_ignoring_nans(y_true, y_estimated):
    return mean_absolute_error(y_true=y_true, y_estimated=y_estimated)


@nan_ignoring
def mean_absolute_percentage_error_ignoring_nans(y_true, y_estimated):
    return mean_absolute_percentage_error(y_true=y_true, y_estimated=y_estimated)


@nan_ignoring
def neg_mean_absolute_percentage_error_ignoring_nans(y_true, y_estimated):
    return -mean_absolute_percentage_error(y_true=y_true, y_estimated=y_estimated)


@nan_ignoring
def mean_adjusted_absolute_percentage_error_ignoring_nans(y_true, y_estimated):
    return mean_adjusted_absolute_percentage_error(y_true=y_true, y_estimated=y_estimated)


@nan_ignoring
def mean_absolute_scaled_error_ignoring_nans(y_train, y_true, y_estimated, seasonal_period=1):
    return mean_absolute_scaled_error(y_train=y_train, y_true=y_true, y_estimated=y_estimated,
                                      seasonal_period=seasonal_period)
