# This file containing two ways to construct time series analysis models,
# a naive model without hyperparameter tuning, and a tuned model.

import pandas as pd
import numpy as np
from prophet import Prophet
import itertools
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from tqdm import tqdm

def get_naive_model(data: pd.DataFrame):
    '''
    Get a naive model without any hyperparameter tuning and fit the model to the data.
    ---
    Params:
    data: Pandas DataFrame containing time column 'ds' and a data column 'y'
    ---
    Output:
    A naive Prophet model
    '''

    model = Prophet()
    model.fit(data)

    return model

def get_tuned_model(data: pd.DataFrame, hyperparams: dict, cutoffs: pd.date_range, horizon: int):
    '''
    Get a tuned model using grid search. Here is a list of recommended hyperparams to be tuned, you can
    also refer to https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning:
    1. `changepoint_prior_scale`: determine how large the trend change can be at changepoints.
    The default value for this hyperparameter is 0.05. I recommend to try 0.01 to 0.1.
    2. `yearly_seasonality`: determine the order of Fourier Series used in modeling the yearly seasonality.
    The default value for this hyperparameter is 10 (if the data is longer than 1 year). And will not be
    turned on (if the data is less than 1 year).
    A larger value will result in better fit, but may lead to overfitting. I recommend to try 5 to 20.
    3. `weekly_seasonality`: Similar to yearly_seasonality. The default value for this hyperparameter is 3.
    I recommend to try 1 to 5.
    4. `daily_seasonality`: Similar to yearly_seasonality. But raraly used.
    5. `changepoint_range`: The percentage of training time period for the model to insert change point.
    The default value for this hyperparameter is 0.8. You can try 0.7 to 0.9 based on the shape of data.
    6. Other parameters suggested to be tuned can be found in the link.

    **Note this function make takes hours to run!**
    ---
    Params:
    data: Pandas DataFrame containing time column 'ds' and a data column 'y'
    hyperparams: a dict containing hyperparameters to be tuned as keys and a list of possible points as
    values.
    cutoffs: pd.date_range indicating the training time range of the data, e.g. pd.date_range(start='2017-02-01', end='2022-05-01', freq='MS')
    horizon: the time range left as validation set, unit is days.
    ---
    Output:
    A tuned Prophet model
    '''
    # Generate all combinations of parameters
    all_params = [dict(zip(hyperparams.keys(), v)) for v in itertools.product(*hyperparams.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in tqdm(all_params):
        m = Prophet(**params).fit(data)  # Fit model with given params
        df_cv = cross_validation(m, cutoffs = cutoffs, horizon = str(horizon) + ' days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window = 1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses

    # get the best parameters
    best_params = all_params[np.argmin(rmses)]

    # create that model and fit the data
    best_model = Prophet(**best_params)
    best_model.fit(data)

    return best_model


    


