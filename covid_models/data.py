import numpy as np
import pandas as pd

from datetime import datetime
from typing import List, Tuple

from .dynamic_bias import AbstractDynamicBias

_NPIS = ['npi_school_closing',
         'npi_workplace_closing',
         'npi_cancel_public_events',
         'npi_gatherings_restrictions',
         'npi_close_public_transport',
         'npi_stay_at_home',
         'npi_internal_movement_restrictions',
         'npi_international_travel_controls',
         'npi_public_information',
         'npi_testing_policy',
         'npi_contact_tracing',
         'npi_stringency_index',
         'npi_masks']

_MOBILITY = ['mobility_retail_recreation',
             'mobility_grocery_pharmacy',
             'mobility_parks',
             'mobility_transit_stations',
             'mobility_workplaces',
             'mobility_residential',
             'mobility_travel_driving',
             'mobility_travel_transit',
             'mobility_travel_walking']


def _create_controls(x: pd.Series,
                     features: np.ndarray,
                     weeks_back=4,
                     future_days=7,
                     alpha=1.0):
    """Create weeks_back controls."""
    # Ensure that we have a long enough history of features, so that for any
    #  `x` we can create a features vector (or control vector).
    days_back = weeks_back * 7
    zero_features = np.zeros((days_back, features.shape[1]))
    features = np.append(zero_features, features, axis=0)

    delta = features.shape[0] - x.shape[0]
    controls = []
    for idx in range(x.shape[0]):
        control = []
        for offset in range(days_back, 0, -7):
            start = delta + idx - offset
            week_average_control = np.mean(features[start:start + 7, :],
                                           axis=0, keepdims=True)
            control.append(week_average_control)

        # Concatenate list of weekly average controls.
        # control shape: (weeks_back, num_controls).
        control = np.concatenate(control, axis=0)

        # Flatten the control to form a control vector.
        # control shape: (1, weeks_back * num_controls).
        control = np.reshape(control, (1, -1))
        controls.append(control)

    controls = np.concatenate(controls, axis=0)
    # The controls have shape (past_time, weeks_back * num_controls).

    future_controls = []
    last_feature = alpha * features[-1, :]

    future = np.tile(last_feature, (future_days, 1))
    features = np.append(features, future, axis=0)

    for idx in range(future_days):
        control = []
        for offset in range(days_back, 0, -7):
            start = delta + x.shape[0] + idx - offset
            week_average_control = np.mean(features[start:start + 7, :],
                                           axis=0, keepdims=True)
            control.append(week_average_control)

        control = np.concatenate(control, axis=0)
        control = np.reshape(control, (1, -1))
        future_controls.append(control)

    future_controls = np.concatenate(future_controls, axis=0)
    # The future_controls have shape (future_time, weeks_back * num_controls).

    if np.isnan(controls).any():
        return None, None

    return controls, future_controls


def _get_range(x: pd.Series):
    """Get a range of values so that there are no NaNs in the sequence."""

    first_idx = x.first_valid_index()
    last_idx = x.last_valid_index()
    subset = x.loc[first_idx:last_idx]

    while subset.isnull().values.any() and \
            (first_idx is not None or last_idx is not None):
        idx = subset.isna().idxmax()
        first_idx = subset.loc[idx:last_idx].first_valid_index()
        subset = x.loc[first_idx:last_idx]

    if first_idx is None or last_idx is None:
        return None, None

    return first_idx, last_idx


class CovidData(object):
    def __init__(self, remove_china_correction: bool=False):
        # Read in the data.
        df = pd.read_csv(
            'https://raw.githubusercontent.com/rs-delve/covid19_datasets'
            '/master/dataset/combined_dataset_latest.csv')
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
        df = df.set_index(['ISO', 'DATE'])

        if remove_china_correction:
            adjustment_date = datetime(2020, 4, 17)
            df.at[('CHN', adjustment_date), 'deaths_new'] = 0

        for npi in _NPIS:
            df[npi] = df[npi] / df[npi].max()

        for npi in _MOBILITY:
            df[npi] = df[npi] / 100.0

        # Convert cases, deaths and tests to be rolling 7 day values
        df['rolling_cases'] = df['cases_new'].copy()
        df['rolling_deaths'] = df['deaths_new'].copy()
        df['rolling_tests'] = df['tests_new'].copy()

        rolling_cols = ['rolling_cases', 'rolling_deaths', 'rolling_tests']
        df.loc[:, rolling_cols] = df.loc[:, rolling_cols]. \
            groupby(level=0). \
            apply(lambda rows: rows.rolling(7).sum())

        df['deaths_week_avg'] = df['rolling_deaths'].copy() / 7.0

        df['mobility'] = (df['mobility_retail_recreation'] +
                          df['mobility_grocery_pharmacy'] +
                          df['mobility_parks'] +
                          df['mobility_transit_stations'] +
                          df['mobility_workplaces'] -
                          df['mobility_residential'])
        df.loc[:, ['mobility']] = df.loc[:, ['mobility']]. \
            groupby(level=0). \
            apply(lambda rows: rows.rolling(7).sum())
        df['mobility'] = df['mobility'] / 7.0

        self.df = df.replace([np.inf, -np.inf], np.nan)

    def countries_with_cumulative_mortality_greater_than(
            self,
            min_cumulative_mortality: int,
            date: datetime) -> List[str]:
        """Get a list of countries with cumulative mortality greater than
        `min_cumulative_mortality` on `date`.
        """

        cumulative_mortality = self.df.xs(date,
                                          level='DATE',
                                          drop_level=False)['deaths_total']
        isos = cumulative_mortality[
            cumulative_mortality > min_cumulative_mortality]
        return isos.index.get_level_values('ISO').unique().tolist()

    def create_stringency_vectors(
            self,
            iso: str,
            days_back_list: List=None,
            max_date: datetime=None) -> Tuple[pd.DatetimeIndex,
                                              np.ndarray,
                                              np.ndarray]:
        """Constructs Oxford stringency vectors for each day for a country.

        Returns:
            t: The time index of the data
            y: A vector of average mortality per day
            features: A matrix of Oxford stringency index features for each day
                (as rows)
        """

        if days_back_list is None:
            days_back_list = [14, 21, 28]

        df = self.df.loc[iso].copy()
        if max_date is not None:
            df = df.loc[:max_date]

        df['rolling_stringency'] = df['npi_stringency_index'].copy()

        df.loc[:, ['rolling_stringency']] = \
            df.loc[:, ['rolling_stringency']].apply(
            lambda rows: rows.rolling(7).sum())
        df['rolling_stringency'] = df['rolling_stringency'] / 7.0

        def week_str(col_name: str, week: int):
            return col_name + '_' + str(week)

        col_list = []
        for days_back in range(days_back_list):
            col = week_str('rolling_stringency', week)
            df[col] = df['rolling_stringency'].copy()
            df[col] = df[col].shift(days_back)
            df[col] = df[col].fillna(value=0)
            df[col] = df[col].fillna(method='ffill')
            col_list.append(col)

        t = df.index
        y = df['deaths_week_avg'].fillna(value=0).to_numpy()
        features = df[col_list].to_numpy()

        return t, y, features

    def get_country_data(self,
                         iso: str,
                         npis: List[str],
                         exclude_weekly_average_below: float=1.0,
                         max_date: datetime=None,
                         weeks_back: int=4,
                         global_bias: bool=True,
                         country_bias: bool=False,
                         country_index: int=0,
                         total_countries: int=0,
                         dynamic_bias: AbstractDynamicBias=None,
                         alpha: float=1.0,
                         future_days: int=7) -> Tuple[pd.Series,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      pd.DataFrame]:
        """Get the logarithm of the weekly average COVID-19 deaths for each
        day for a country, with a feature vector for each day.

        Args:
            iso: The country ISO code.
            npis: A list of non-pharmaceutical interventions to include in the
                feature vector for each day.
            exclude_weekly_average_below: The minimum weekly average COVID-19
                deaths for including in the time series. We assume the time
                series is too noisy below the minimum we specify. The default
                minimum is 1.0.
            max_date: If specified, data is truncated at `max_date`. This is
                useful if we want to do comparisons on future predictions.
            weeks_back: The number of weeks of non-pharmaceutical interventions
                (NPIs) that should be included in a feature vector. For every
                week before a day (i.e. last 1-7 days ago, 8-14 days ago, etc.)
                for every NPI, the average NPI for the week is a feature.
            global_bias: If true, the feature vector will include a bias
                component of "1".
            country_bias: If true, the feature vector will contain a "one hot"
                encoding of the country index. If it is true, both
                `country_index` and `total_countries` need to be set.
            country_index: The one-hot encoding index of the country, required
                for when `country_bias` is true.
            total_countries: The total number of countries, required for the
                length of the one-hot encoding of a country when `country_bias`
                is true.
            dynamic_bias: An instance of `AbstractDynamicBias`, which handles
                the processing of dynamic (recurrent) bias features (controls).
            alpha: A non-negative value which indicates how much the last day's
                non-pharmaceutical interventions (NPIs) are scaled to create
                a feature vector for future days. A value of 0.0 means that
                all NPIs are set to 0; a value of 1.0 means that NPIs continue
                as on the last day.
            future_days: The number of days in the future for which feature
                vectors will be created.

        Returns:
            x: A pandas Series, which contains the logarithm of the weekly
                average COVID-19 deaths as a time series. It is created so that
                there are no NaNs or negative infinite values in the time
                series. Small disconnected snippets early on in the time
                series, like when there is a single fatal COVID-19 case in a
                month, are trimmed. In cases where the original data was
                [NaN, a, NaN, b, c, d], the series [b, c, d] is returned.
            controls: A matrix of feature vectors. Each row contains a
                feature vector for the corresponding element in `x`.
            future_controls: A matrix of hypothesized future feature vectors.
            country_df: The data frame for country with ISO code `iso`.

        Remarks:
            The controls on day t-1 are intended to effect day t, and the
            dynamic bias term is constructed that way. It is intended to be
            used in a Gaussian Linear Dynamical System which has the form:
            z_t | z_{t-1}, u_{t-1} = ...
            y_t | z_t = ...
            where latent state z_t depends on the latent state z_{t-1} and
            the previous day's controls or features u_{t-1}. y_t is the
            observation on day t.
        """

        country_df = self.df.loc[iso]
        if max_date is not None:
            country_df = country_df.loc[:max_date]

        x = country_df['deaths_week_avg']

        # We assume that data is too noisy when mortality is less than
        # `exclude_weekly_average_below` per day.
        x[x < exclude_weekly_average_below] = np.nan
        x = np.log(x)
        x = x.replace([np.inf, -np.inf], np.nan)

        # Get the range that we'll use for the time series.
        first_idx, last_idx = _get_range(x)

        if first_idx is None:
            return None, None, None, None
        else:
            x = x.loc[first_idx:last_idx]

            # Get the starting date for policies.
            features = country_df[npis]

            # Sometimes NPI data lags a day. Forward fill them.
            features = features.fillna(method='ffill')
            features_first_idx = features.first_valid_index()
            features = features.loc[features_first_idx:last_idx].to_numpy()

            controls, future_controls = _create_controls(
                x, features, weeks_back=weeks_back, future_days=future_days,
                alpha=alpha)

            if controls is None:
                return None, None, None, None

            if global_bias:
                ones = np.ones((controls.shape[0], 1))
                controls = np.append(controls, ones, axis=1)
                ones = np.ones((future_controls.shape[0], 1))
                future_controls = np.append(future_controls, ones, axis=1)

            if country_bias:
                indicator_column = np.zeros((controls.shape[0],
                                             total_countries))
                indicator_column[:, country_index] = 1
                controls = np.append(controls, indicator_column, axis=1)
                indicator_column = np.zeros((future_controls.shape[0],
                                             total_countries))
                indicator_column[:, country_index] = 1
                future_controls = np.append(future_controls, indicator_column,
                                            axis=1)

            if dynamic_bias is not None and dynamic_bias.dim > 0:
                dyn_bias = dynamic_bias.get_dynamic_bias_from_df(x, country_df)

                # Append the dynamic biases to the controls, but not the future
                # controls.
                controls = np.append(controls, dyn_bias, axis=1)

        return x, controls, future_controls, country_df

    def get_x(self,
              iso: str,
              first_day: datetime,
              last_day: datetime) -> pd.Series:
        country_df = self.df.loc[iso]
        country_df = country_df.loc[first_day:last_day]
        x = np.log(country_df['deaths_week_avg'])
        x = x.replace([np.inf, -np.inf], np.nan)

        return x
