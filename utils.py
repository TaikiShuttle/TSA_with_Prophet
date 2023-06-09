# In this file, there are utility fuctions that can be used to do time series analysis
import pandas as pd
import numpy as np
from prophet import Prophet
from get_models import *
from prophet.plot import add_changepoints_to_plot
from prophet.serialize import model_to_json, model_from_json

def read_data(filename: str, filetype: str = 'csv'):
    '''
    Read in data from either an excel file or a csv file.
    ---
    Params:
    filename: name of the file
    filetype: type of the file, either csv or excel
    ---
    Output:
    A Pandas dataframe
    '''
    if filename == 'csv':
        return pd.read_csv(filename)
    
    elif filename == 'excel':
        return pd.read_excel(filename)
    
    else:
        return TypeError("file type not in csv or excel!")
    

def change_colname(df: pd.DataFrame, time_col: str, data_col: str):
    '''
    Change the column names of input dataframe in order to use Prophet.
    ---
    Params:
    df: input dataframe
    time_col: name of the time column in the dataframe
    data_col: name of the data column in the dataframe
    ---
    Output:
    Modified Pandas dataframe, only containing time column 'ds' and data column 'y'
    '''
    new_df = df[[time_col, data_col]]
    new_df.columns = ['ds', 'y']

    return new_df

def make_future_df(model: Prophet, periods: int, freq: str):
    '''
    Generate a dataframe both containing the training data and the times for prediction.
    ---
    Params:
    model: The model used in training, it should have already been fitted to a dataset
    periods: how long should the model predict
    freq: what is the sampling frequency. day(D)? week? month(M or MS)? specify using Pandas style
    ---
    Output:
    A Pandas DataFrame containing a 'ds' column indicating the total range, including training and predicting.
    '''

    future = model.make_future_dataframe(periods = periods, freq = freq)

    return future


def make_forecast(model: Prophet, future: pd.DataFrame):
    '''
    Make predictions using model.
    ---
    Params:
    model: prediction model
    future: a Pandas DataFrame containing 'ds', indicating the prediction range. This usually contains
    both the training and the future prediction part. 
    ---
    Output:
    A Pandas DataFrame containing prediction results
    '''

    forecast = model.predict(future)
    return forecast

def plot_forecast_results(model: Prophet, forecast: pd.DataFrame):
    '''
    Make a plot for the predictions both for training part and prediction part.
    ---
    Params:
    model: model used in forecasting
    forecast: a Pandas DataFrame containing prediction results
    ---
    Output:
    A plot instance.
    '''

    # we are majorly interested in the following parameters.
    fig = model.plot(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    return fig

def make_forecast_numpy(forecast: pd.DataFrame):
    '''
    Return an forecast array in four dimensions, 'ds', 'yhat', 'yhat_lower', 'yhat_upper', for frontend.
    ---
    Params:
    forecast: A numpy.array containing prediction results
    '''
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_numpy()

def save_model(model: Prophet, filename: str):
    '''
    Save an existing model.
    ---
    Params:
    model: model to be saved
    filename: target file name
    ---
    Output:
    None
    '''
    with open(filename, 'w') as f:
        f.write(model_to_json(model))

    return

def load_model(filename: str):
    '''
    Load a model from file.
    ---
    Params:
    filename: name of source file
    ---
    Output:
    A loaded model
    '''
    with open(filename, 'r') as f:
        return model_from_json(f.read())
    
def update_model(data: pd.DataFrame, original_model: Prophet):
    '''
    Retrieve parameters from a trained model, and update the model using new dataset.
    ---
    Params:
    data: A new Pandas DataFrame used to update the model, containing 'ds' and 'ys'
    original_model: original trained model
    ---
    Output:
    An updated model
    '''

    # load in the original parameters
    parameters = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if original_model.mcmc_samples == 0:
            parameters[pname] = original_model.params[pname][0][0]
        else:
            parameters[pname] = np.mean(original_model.params[pname])
    for pname in ['delta', 'beta']:
        if original_model.mcmc_samples == 0:
            parameters[pname] = original_model.params[pname][0]
        else:
            parameters[pname] = np.mean(original_model.params[pname], axis=0)

    # update the model using the data
    model = Prophet().fit(data, init = parameters)
    
    return model


