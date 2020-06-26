import abc
import numpy as np
import pandas as pd

from typing import List


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
    """A feature vector with elements x_t - x_{t - i}, where x_t is the
    logarithm of the average COVID-19 deaths for the week ending in day t."""

    def __init__(self, days_back: List[int]):
        for d in days_back:
            assert d > 0
        self.days_back = days_back

    @property
    def dim(self):
        return len(self.days_back)

    def get_dynamic_bias_from_df(self, x, country_df):
        x_days_back = []
        for d in self.days_back:
            x_back = x.shift(periods=d).fillna(method='backfill')
            x_days_back.append(x_back.to_numpy())

        return self._get_dynamic_bias(x.to_numpy(),
                                      x_days_back)

    def _get_dynamic_bias(self, x, x_days_back):
        deltas = [np.reshape(x - y, (-1, 1)) for y in x_days_back]
        return np.concatenate(deltas, axis=1)

    def get_last_dynamic_bias(self, x, country_df):
        x_days_back = [x[- 1 - d] for d in self.days_back]
        return self._get_dynamic_bias(x[-1], x_days_back)


class DynamicBiasDeltasQuadratic(AbstractDynamicBias):
    """A feature vector with elements x_t - x_{t - i} and (x_t - x_{t - i})^2,
    where x_t is the logarithm of the average COVID-19 deaths for the week
    ending in day t."""

    def __init__(self, days_back: List[int]):
        for d in days_back:
            assert d > 0
        self.days_back = days_back

    @property
    def dim(self):
        return 2 * len(self.days_back)

    def get_dynamic_bias_from_df(self, x, country_df):
        x_days_back = []
        for d in self.days_back:
            x_back = x.shift(periods=d).fillna(method='backfill')
            x_days_back.append(x_back.to_numpy())

        return self._get_dynamic_bias(x.to_numpy(),
                                      x_days_back)

    def _get_dynamic_bias(self, x, x_days_back):
        deltas = [np.reshape(x - y, (-1, 1)) for y in x_days_back]
        deltas2 = [np.reshape((x - y) * (x - y), (-1, 1)) for y in x_days_back]
        return np.concatenate(deltas + deltas2, axis=1)

    def get_last_dynamic_bias(self, x, country_df):
        x_days_back = [x[- 1 - d] for d in self.days_back]
        return self._get_dynamic_bias(x[-1], x_days_back)


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


class DynamicBiasLogRemainder(AbstractDynamicBias):
    """The logarithm of (1 - normalized cumulative mortality) up to day t."""

    def __init__(self):
        self.dbc = DynamicBiasCumulative()

    @property
    def dim(self):
        return self.dbc.dim

    def get_dynamic_bias_from_df(self, x, country_df):
        y = self.dbc.get_dynamic_bias_from_df(x, country_df)
        return np.log(np.maximum(1 - y, 1e-1))

    def get_last_dynamic_bias(self, x, country_df):
        y = self.dbc.get_last_dynamic_bias(x, country_df)
        return np.log(np.maximum(1 - y, 1e-1))


class DynamicBiasArray(AbstractDynamicBias):
    """The logarithm of (1 - normalized cumulative mortality) up to day t."""

    def __init__(self, dynamic_biases: List[AbstractDynamicBias]):
        self.dynamic_biases = dynamic_biases

    @property
    def dim(self):
        dims = [db.dim for db in self.dynamic_biases]
        return sum(dims)

    def get_dynamic_bias_from_df(self, x, country_df):
        dbs = [db.get_dynamic_bias_from_df(x, country_df)
               for db in self.dynamic_biases]
        return np.concatenate(dbs, axis=1)

    def get_last_dynamic_bias(self, x, country_df):
        dbs = [db.get_last_dynamic_bias(x, country_df)
               for db in self.dynamic_biases]
        return np.concatenate(dbs, axis=1)
