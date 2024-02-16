

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as pt
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2020-01-01' #yyyy-mm-dd
end = '2024-2-8'   #yyyy-mm-dd

st.title('Stock Trend prediction')
user_input = st.text_input('Enter Stock Ticker','IRFC.NS')
df = yf.download(user_input, start=start, end=end)


#decribing data
st.subheader('Data from 2020-2024')
st.write(df.describe()) 


#visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

 # spliting data into Training and Testing
Data_training = pd.DataFrame(df['Close'][0:int(len(df)* 0.70)])
Data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(Data_training.shape)
print(Data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

Data_training_array = scaler.fit_transform(Data_training)


#load model
model = load_model('iolcp_keras_model.h5')


#testing part
past_100_days = Data_training.tail(100)

final_df = pd.concat([past_100_days,Data_testing],ignore_index= True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)

y_predicated = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor


#final grap
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label ='original price')
plt.plot(y_predicated,'r',label ='Predicated price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)