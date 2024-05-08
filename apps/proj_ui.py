import streamlit as st
import datetime as dt
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd 
import pandas_datareader.data as web
import numpy as np 
from matplotlib import style
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.svm import SVR

def app():
    end = dt.datetime.now()
    start = end - dt.timedelta(days=1825)
    start, end

    stocklist=['MSFT']
    df = web.DataReader(stocklist,'yahoo', start, end)

    st.title("Stock Prediction")
    st.write("###")

    n_years = st.slider("", 1, 5)
    period = n_years * 365

    st.write("###")

    st.subheader("Raw data")
    st.write(df.tail())


    import matplotlib.dates as mdates
     
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    org_dates = dates_df['Date']

    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_df['Date'], y=df['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=dates_df['Date'], y=df['Close'], name='stock_close'))
        fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
        st.plotly_chart(fig)

    plot_raw_data()


    dates = dates_df['Date'].to_numpy()
    prices = df['Adj Close'].to_numpy()

    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_rbf.fit(dates, prices)

    train_data = df.loc[:,'Adj Close'].to_numpy()
    print(train_data.shape)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)

    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Create the data to train our model on:
    time_steps = 36
    X_train, y_train = create_dataset(train_data, time_steps)

    # reshape it [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 36, 1))

    print(X_train.shape)


    # Visualizing our data with prints: 
    print('X_train:')
    print(str(scaler.inverse_transform(X_train[0])))
    print("\n")
    print('y_train: ' + str(scaler.inverse_transform(y_train[0].reshape(-1,1)))+'\n')

    # Build the model 
    model = keras.Sequential()

    model.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(units = 1))

    # Compiling the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the model to the Training set
    history = model.fit(X_train, y_train, epochs = 20, batch_size = 10, validation_split=.20)

    test_data = df['Adj Close'].values
    test_data = test_data.reshape(-1,1)
    test_data = scaler.transform(test_data)

    # Create the data to test our model on:
    time_steps = 36
    X_test, y_test = create_dataset(test_data, time_steps)

    # store the original vals for plotting the predictions 
    y_test = y_test.reshape(-1,1)
    org_y = scaler.inverse_transform(y_test)

    # reshape it [samples, time steps, features]
    X_test = np.reshape(X_test, (X_test.shape[0], 36, 1))

    # Predict the prices with the model
    predicted_y = model.predict(X_test)
    predicted_y = scaler.inverse_transform(predicted_y)


    # plot the results 
    st.write("***")
    st.write("###")

    st.subheader("Forecast data")

    fig1 = plt.figure(figsize=(12,6))
    plt.plot(org_y, color = 'red', label = 'Real Microsoft Stock Price')
    plt.plot(predicted_y, color = 'blue', label = 'Predicted Microsoft Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Microsoft Stock Price')
    plt.legend()
    st.plotly_chart(fig1)


