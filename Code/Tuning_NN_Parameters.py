#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import sqrt
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, output_file, show
from bokeh.models import Legend
output_notebook()

import warnings

# Sklearn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# LSTM Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Custom Package Written to Scape Webdata
from Web_Scrape import scrape_data


# Beyond these libraries, let's also load in some functions we previously wrote to help us transform our data.

# In[2]:


def shift_timeseries(data: pd.DataFrame, lag: int=1):
    """
        Takes in a single column and shifts the data down by the
        number from 'lag.' Places zero in new spot.

        Input:
            data: Dataframe of a single column
            lag: Number of steps to shift data
        Returns:
            Dataframe with orignal data and shifted data
    """
    df = pd.DataFrame(data)

    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.columns = ["Predict Price", "Price"]
    df.fillna(0, inplace=True)
    return df


# ## Load in Data
#
# First load in and confirm that we have the right data.

# In[3]:


model_data = pd.read_csv("Data/Stoneforge_Mystic_transform_data.csv")

model_data.head()


# Unfortunately, our index from our transformed dataset was reset to the first column.  So let's quickly set the index back to the datetime stamp.

# In[4]:


model_data.columns = ["Date", "Predict Price", "Price"]
model_data["Date"] = pd.to_datetime(model_data["Date"])
model_data.index = model_data['Date']

model_data = model_data.drop("Date", axis=1)

model_data.head()

# In[5]:


data = pd.read_csv("Data/Stoneforge_Mystic.csv")

# Move the date column to the index
data["Date"] = pd.to_datetime(data["Date"])
data.index = data["Date"]

data = data.drop("Date", axis=1)


# In[6]:


def custom_test_train(data: np.ndarray, percentage: int):
    """
        Split the time series data into training and test data based first x percentage of data
        returns train and test which are arrays

        Inputs:
            data - 1-d Array of data we want to split
            pecentage - First x percentage of data we want to get
        Returns:
            train - First x percentage of data as an array
            test - Last 100-x percentage of data as an array
            break_point - index of the date where the array was divided
    """

    percent = percentage/100
    break_point = round(len(data)*percent)
    train = data[0:break_point]
    test = data[break_point:]

    return train, test, break_point+1


# In[7]:


model_values = model_data.values

# Split our data into 50% training data and 50% testing data
train, test, split_index = custom_test_train(model_values, 50)

# In[10]:


def scale(temp_train: np.ndarray, temp_test: np.ndarray):
    '''
        Intakes training/testing data and then scales them using a MinMaxScaler.

        Inputs:
            train - Array of training data
            test - Array of testing data
        Returns:
            scaler - scaling method used
            train_scaled - scaled training data
            test_scaled - scaled testing data
    '''
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(temp_train)
    # transform train
    temp_train = train.reshape(temp_train.shape[0], temp_train.shape[1])
    train_scaled = scaler.transform(temp_train)
    # transform test
    temp_test = temp_test.reshape(temp_test.shape[0], temp_test.shape[1])
    test_scaled = scaler.transform(temp_test)
    return scaler, train_scaled, test_scaled


# In[11]:

scaler, train_scaled, test_scaled = scale(train, test)


# In[12]:


def fit_lstm(train: np.ndarray, n_lag:int, n_batch: int, nb_epoch: int, n_neurons: int):
    """
        Fits and trains a single layer LSTM model to the training data

        Inputs:
            train - Training data to fit the model to
            n_lag - Number of days our data is lagging
            n_batch - Amount of training data passed into the model before updating
            nb_epoch - Number of times going completely through the dataset
            n_neurons - Number of neurons in each layer
        Returns:
            model - LSTM model with Sequential() attributes
                Documentation: https://keras.io/models/sequential/
    """
    # Split the data into features and target variable
    X, y = train[:, 0:n_lag], train[:,n_lag:] # X is our features, y is our target variable
    # [samples, timesteps, features]

    # Reshape our feature data so it's in the proper shape for the LSTM model
    X = X.reshape(X.shape[0],1, X.shape[1])
    # Surpress deprecation warnings
    warnings.filterwarnings("ignore")

    # Create the network
    model = Sequential()
    # Add the LSTM model layer
    model.add(LSTM(n_neurons, batch_input_shape = (n_batch, X.shape[1],
                                                   X.shape[2]), stateful=True))

    # Add the output layer
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # Print out every 5 epochs so we known where the model is
        if (i+1)%5 == 0:
            print('Epoch No. '+str(i+1)+'/'+str(nb_epoch))
        model.fit(X,y, epochs = 1, batch_size=n_batch, verbose=0,
                  shuffle=False)

        # Want to make sure each state is reset at the end of a training
        model.reset_states()
    return model



# In[13]:
magic_model = fit_lstm(train_scaled, n_lag=1, n_batch=1,
                       nb_epoch=50, n_neurons=4)


# In[14]:


def forecast_lstm(model, batch_size: int, X: np.ndarray):
    """
        Reshapes input data and predicts based on the input trained model

        Inputs:
            model - trained LSTM model
            batch_size - int, number of values we want to use to predict forward
            X - array of floats, training data
        Returns:
            yhat - array of floats, predicted value
    """
    X = X.reshape(1, 1, len(X)) # Shape the data so it can be used in the LSTM
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

def invert_scale(scaler, X: np.ndarray, value: np.ndarray):
    """
        Does the inverse of whatever scaler is passed in

        Inputs:
            scaler - scaler oringally used
            X - array of floats, Orignal data to adjust
            value - array of floats, how to change the data
        Returns:
            inverted - array of floats, input value that has been inverted
    """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def inverse_difference(history, yhat: np.ndarray, interval: int=1):
    """
        Adds the prediction value to the orginal data values

        Inputs:
            history - orginal data
            yhat - array of floats, difference to add
            interval - int, index of original data to add the number to
        Returns:
            Orginal data plus the difference.
    """
    return yhat + history[-interval]


# Now we can put all these functions in a loop that will predict forward one value at a time.  We'll create a log that we can look at individual RMSE per day and print out a bit of that log to see where we’re at.

# In[15]:


predictions = list()
prediction_log = list() # Log of predictions with expected values
prediction_stats = list() # Keep track of predicted, expected, and RMSE numbers
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i,-1]
    yhat = forecast_lstm(magic_model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(data.values, yhat, len(test_scaled)+1-i)

    predictions.append(yhat)
    expected = data.values[len(train)+ i + 1] #  Go from the end of the training values as the actual values

    RMSE = sqrt(mean_squared_error(expected, yhat))
    temp_stats = np.array([i+1, yhat, expected, RMSE])
    prediction_stats.append(temp_stats)
    prediction_log.append('Day=%d, Predicted=$%f, Expected=$%f, RMSE=$%f' % (i+1, yhat, expected, RMSE))


for i in range(10):
    print(prediction_log[i])


# Notice that the model outputs "Day=#" in the first column which is not a great and we would rather have it as an actual datetime.  We can use the split point from when we split our data into training and testing data to find the actual dates that correlate to those day numbers.

# In[16]:


split_date = data.index[split_index]
predictions_df = pd.DataFrame(prediction_stats,
                              columns=['Date', 'Predicted', 'Expected', 'RMSE'])

# Switch the days to the date
predictions_df["Date"] = data.index[split_index:]







