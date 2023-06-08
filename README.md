# TSA_with_Prophet
A template for time series analysis using Prophet procedure. This is used for potential online deployment for Digital Audit Studio of @Huayun IT Co., Ltd. 

### Intuition
One of the major challenges in time series analysis is overcoming the lagging problem. Traditional time series analysis algorithms like ARIMA often respond to sudden changes slowly, i.e., takes several time units to reflect the change. Meanwhile, audit data prediction requires a instant prediction that can reflecting a proper trend. <a href="https://facebook.github.io/prophet/">Prophet</a> developed by facebook provides a relatively accurate prediction procedures for time series. It performs almost equally well when the data has large variations. Based on this strength, I decided to write some functions that encapsulate the model selection and tuning process of Prophet. It contains a little bit my own understanding of the algorithm as well. And this repo might be a starting point for potential deployment develpoment.

### Requirements
Pandas >= 1.5.3

Numpy >= 1.22.4

Prophet >= 1.1.3

### How to use
##### Build new model
To come up with a complete time series analysis, here is a procedure:
1. call `read_data` function to read in data, and call `change_colname` to modify the column names
2. get a model, using either `get_naive_model` or `get_tuned_model`
3. determine the prediction range, using `make_future_df` function to generate the DataFrame
4. make forecasting using function `make_forecast`
5. try to plot the forecasting results `plot_forecast_results` and see the results
6. Save the model using `save_model`

##### Update model iteratively
When model is built, it should be updated iteratively as new data coming in:
1. call `read_data` to read in the new data
2. load the original saved model using `load_model`
3. call function `update_model`
4. determine the prediction range, using `make_future_df` function to generate the DataFrame
5. make forecasting using function `make_forecast`
6. try to plot the forecasting results `plot_forecast_results` and see the results
7. have a look at the plot, is there a necessity to go over the hyperparameter tuning again?
8. Save the model using `save_model`

### Future Development
Future development of this project includes considering deployment. Since the most deployments are using Java, it might be necessary to translate this project to Java.


