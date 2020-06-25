import abc
import numpy as np
import pandas as pd


class AbstractDynamicBias(abc.ABC):
    """An external control that depends on past observations."""

    @abc.abstractproperty
    def dim(self) -> int:
        """The number of components of the extra control vector."""

    @abc.abstractmethod
    def get_dynamic_bias_from_df(self, x: pd.Series,
                                 country_df: pd.DataFrame) -> np.ndarray:
        """Get a dynamic bias feature vector for every data point in x.

        Args:
            x: A series containing the logarithm of the week-averaged COVID-19
                deaths, for the seven-day period ending on each day in the
                series.
            country_df: A dataframe corresponding to the time series, from
                which other statistics could be drawn.
        """

    @abc.abstractmethod
    def get_last_dynamic_bias(self, x: np.ndarray,
                              country_df: pd.DataFrame) -> np.ndarray:
        """Get a dynamic bias feature vector for the last day in x."""


class DynamicBiasDeltas(AbstractDynamicBias):
    """A 3-dimensional feature vector with elements
            [x_t - x_{t - 3}, x_t - x_{t - 7}, x_t - x_{t - 10}]
    where x_t is the logarithm of the average COVID-19 deaths for the week
    ending in day t."""

    def __init__(self):
        self._dim = 3

    @property
    def dim(self):
        return self._dim

    def get_dynamic_bias_from_df(self, x, country_df):
        x_min_3 = x.shift(periods=+3).fillna(method='backfill')
        x_min_7 = x.shift(periods=+7).fillna(method='backfill')
        x_min_10 = x.shift(periods=+10).fillna(method='backfill')

        return self._get_dynamic_bias(x.to_numpy(),
                                      x_min_3.to_numpy(),
                                      x_min_7.to_numpy(),
                                      x_min_10.to_numpy())

    def _get_dynamic_bias(self, x, x_min_3, x_min_7, x_min_10):
        delta_3 = x - x_min_3
        delta_7 = x - x_min_7
        delta_10 = x - x_min_10

        return np.concatenate((np.reshape(delta_3, (-1, 1)),
                               np.reshape(delta_7, (-1, 1)),
                               np.reshape(delta_10, (-1, 1))),
                              axis=1)

    def get_last_dynamic_bias(self, x, country_df):
        return self._get_dynamic_bias(x[-1], x[-4], x[-8], x[-11])


class DynamicBiasCumulative(AbstractDynamicBias):
    """The population-normalized cumulative mortality up to day t."""

    def __init__(self):
        self._dim = 1

    @property
    def dim(self):
        return self._dim

    def get_dynamic_bias_from_df(self, x, country_df):

        population = country_df['stats_population'].max()

        y = np.exp(x).cumsum().to_numpy()
        y = 1000 * y / population

        return np.reshape(y, (-1, 1))

    def get_last_dynamic_bias(self, x, country_df):
        population = country_df['stats_population'].max()
        y = np.sum(np.exp(x))
        y = 1000 * y / population

        return np.reshape(y, (-1, 1))

class DynamicBiasLogCumulative(AbstractDynamicBias):
    """The logarithm of the cumulative mortality up to day t.

    Remarks: A constant log normalizer (like the country's total population) is
    not subtracted, as it is a country-specific constant that could be learned
    by a country-specific bias parameter.
    """

    def __init__(self):
        self._dim = 1

    @property
    def dim(self):
        return self._dim

    def get_dynamic_bias_from_df(self, x, country_df):
        y = np.log(np.exp(x).cumsum()).to_numpy()
        return np.reshape(y, (-1, 1))

    def get_last_dynamic_bias(self, x, country_df):
        y = np.log(np.sum(np.exp(x)))
        return np.reshape(y, (-1, 1))
