import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
from streamlit_option_menu import option_menu
import datetime as dt
import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import pandas as pd 
import pandas_datareader.data as web
import numpy as np 
from matplotlib import style
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
from PIL import Image

with st.sidebar:
    choose = option_menu("Navigation", ["Home", "Prediction", "About"],
                         icons=['house', 'kanban', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#50d7f2", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#fc6832"},
    }
    )
logo = Image.open('uitm.png')
profile = Image.open('syafiq.png')
svr_img = Image.open('svr.png')
lstm_img = Image.open('lstm.png')

if choose == "Home":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Home</p>', unsafe_allow_html=True)
      
    with col2:               # To display uitm logo
        st.image(logo, width=130 )

    st.write("\n\n\nWelcome to the Stock Price Prediction Web Application.")
    
    st.write("\n\nThis Web Application is a requirement for CS230 Final Year Project (FYP).")

    st.write("\n\nThe application is build using python programming language and using the Streamlit framework to create the application design.")

    st.write("\n\n\nThe main methods or techniques used in this project is Support Vector Regression (SVR) and Long Short-Term Memory (LSTM).")
    

    st.write("\n\n\n")
    st.image(svr_img)

    st.write("\n\nSVR is used to minimize the error in the dataset and optimize it for training LSTM model.")

    st.image(lstm_img)

    st.write("\n\nLSTM is and advance Recurrent Neural Network due to its capability of storing previous information. This capabilities is usefull to predict the stock price because it look at the stock price for the previous day.")
    
elif choose == "Prediction":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Stock Prediction</p>', unsafe_allow_html=True)
    with col2:               # To display uitm logo
        st.image(logo, width=130 )

        
    end = dt.datetime.now()
    start = end - dt.timedelta(days=1825)
    start, end

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    st.title("Stock Prediction")
    st.write("###")


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')


    st.subheader("Raw data")
    st.write(data.tail())


    

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
        st.plotly_chart(fig)

    plot_raw_data()

    
    train_date = pd.to_datetime(data['Date'])
    dates = data['Date'].to_numpy()
    prices = data['Adj Close'].to_numpy()

    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_rbf.fit(dates, prices)

    st.subheader("Support Vector Regression")
    fig=plt.figure(figsize = (12,6))
    plt.plot(dates, prices, color= 'black', label= 'Data')
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig)

    cols = list(data)[1:6]

    train_data = data[cols].astype(float)
    print(train_data.shape)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    

    scaler.fit(train_data)
    train_data_scale = scaler.transform(train_data)

    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []

    n_future = 7   # Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    
    for i in range(n_past, len(train_data_scale) - n_future +1):
        trainX.append(train_data_scale[i - n_past:i, 0:train_data.shape[1]])
        trainY.append(train_data_scale[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))


    # Build the model 
    model = keras.Sequential()

    model.add(LSTM(units = 64, activation='relu', return_sequences = True, input_shape = (trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(units = 32, activation='relu', return_sequences = False))
    model.add(Dropout(0.2))
    

    # Output layer
    model.add(Dense(trainY.shape[1]))

    # Compiling the model
    model.compile(optimizer = 'adam', loss = 'mse')

    # Fitting the model to the Training set
    history = model.fit(trainX, trainY, epochs = 12, batch_size = 32, validation_split=0.30, verbose=1)

    
    st.subheader("Validation")
    fig, ax = plt.subplots(figsize = (12,6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()
    st.plotly_chart(fig)

    st.write("Training loss: ",history.history['loss'])


    #Predicting...
    #Libraries that will help us extract only business days in the US.
    #Otherwise our dates would be wrong when we look back (or forward).  
    import holidays
    from pandas.tseries.offsets import CustomBusinessDay
    us_bd = CustomBusinessDay(calendar=holidays.MY())
    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    
      #let us predict past 15 days
    
    days = ('7 Days', '2 Weeks', '1 Month', '3 Month', '1 Year')
    selected_days = st.selectbox('Select how long to predict', days)

    if selected_days == '7 Days':
        n_predict = 7

    elif selected_days == '2 Weeks':
        n_predict = 14

    elif selected_days == '1 Month':
        n_predict = 30

    elif selected_days == '3 Month':
        n_predict = 90

    elif selected_days == '1 Year':
        n_predict = 365
    
    predict_period_dates = pd.date_range(list(train_date)[-1], periods=n_predict, freq='1d').tolist()
    print(predict_period_dates)

   

    # Predict the prices with the model
    predicted_y = model.predict(trainX[-n_predict:])
    predicted_copy = np.repeat(predicted_y, train_data.shape[1], axis=-1)
    y_future = scaler.inverse_transform(predicted_copy)[:,0]

    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Close':y_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

    original = data[['Date', 'Close']]
    original['Date']=pd.to_datetime(original['Date'])

    n_end = dt.datetime.now()
    n_start = n_end - dt.timedelta(days=365)
    n_start, n_end

    original = original.loc[original['Date'] > n_start]

    st.subheader("Forecast data")
    fig, ax = plt.subplots(figsize = (12,6)) 
    sns.lineplot(x=original['Date'], y=original['Close'])
    sns.lineplot(x=df_forecast['Date'], y=df_forecast['Close'])
    st.plotly_chart(fig)

    

elif choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)
    with col2:               # To display uitm logo
        st.image(logo, width=130 )

    st.image(profile, width=130 )

    st.write("\n\nMy name is Muhammad Syafiq Bin Ismail Reduan from Group CS2306B. This is my web application for my Final Year Project (FYP) titled Stock Price Prediction Using Support Vector Regression and Long SHort-Term Memory.")

    st.write("\n\nSupervisor: Dr. Ali Bin Seman")
