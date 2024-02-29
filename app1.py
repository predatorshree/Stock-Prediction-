

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as pt
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2010-01-01' #yyyy-mm-dd
end = '2024-02-27'   #yyyy-mm-dd

st.title('Stock Trend prediction')
user_input = st.text_input('Enter Stock Ticker','RELIANCE.NS')
df = yf.download(user_input, start=start, end=end)


#decribing data
st.subheader('Data from 2010-2024')
st.write(df.describe()) 
highest_price = df['Close'].max()
lowest_price = df['Close'].min()
max_volume = df['Volume'].max()

# Displaying the extracted statistics
st.subheader('Specific Statistics')
st.write(f'Highest Price: {highest_price}')
st.write(f'Lowest Price: {lowest_price}')
st.write(f'Maximum Volume: {max_volume}')


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

st.subheader('Closing Price vs Time Chart with EMA(5,10,20)')
# Calculate moving averages
ma5 = df['Close'].rolling(9).mean()
ma10 = df['Close'].rolling(50).mean()
ma20 = df['Close'].rolling(200).mean()

# Plotting
fig = plt.figure(figsize=(12,6))
plt.plot(ma5 , 'r', label='EMA9')
plt.plot(ma10, 'g', label='EMA50')
plt.plot(ma20, 'b', label='EMA200')
plt.plot(df['Close'], 'k', label='Closing Prices')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
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
model = load_model('RELIANCE.NS10.NS_keras_model.h5')


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

# Generate list of dates within the range of start_date and end_date


# Assuming y_predicted is defined


st.subheader('Predictions vs Original')
# Add headline with start date and end date
st.write(f"Start Date: {start}, End Date: {end}")

# Fetch historical stock data for the specified period
stock_data = yf.download(user_input, start=start, end=end)

# Plot original and predicted prices
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original price')
plt.plot(y_predicated, 'r', label='Predicted price')

# Calculate the direction of projection
last_original_price = stock_data.iloc[-1]['Close']
last_predicted_price = y_predicated[-1]
direction = "Upward" if last_predicted_price > last_original_price else "Downward"

# Add text annotation for the direction of projection
plt.text(len(y_test) - 1, last_predicted_price, f'{direction} projection', ha='right', va='center', color='green' if direction == 'Upward' else 'red')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display last predicted value and current price
st.subheader('Last Predicted Value and Current Price')
st.write(f"Last Predicted Value: {last_predicted_price}")
st.write(f"Current Price: {last_original_price}")





st.title('Historical Revenue and Earnings')

# Get user input for stock ticker
#user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Input for start date and end date
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Option selector for frequency
frequency_option = st.selectbox('Select Frequency', ['Quarterly', 'Annual'])

if start_date < end_date:
    # Fetch historical stock data
    stock_data = yf.download(user_input, start=start_date, end=end_date)

    if not stock_data.empty:
        if frequency_option == 'Quarterly':
            # Resample data to quarterly frequency
            resampled_data = stock_data.resample('Q').mean()
        elif frequency_option == 'Annual':
            # Resample data to annual frequency
            resampled_data = stock_data.resample('Y').mean()

        # Calculate Revenue and Earnings in INR
        usd_to_inr_rate = 75.0  # Example rate, you should use the actual current rate
        inr_to_crores = 1e-7  # Conversion factor from INR to crores
        resampled_data['Revenue_INR'] = resampled_data['Close'] * 1000000 * usd_to_inr_rate * inr_to_crores
        resampled_data['Earnings_INR'] = resampled_data['Revenue_INR'] * 0.1  # Assuming earnings as 10% of revenue

        # Plotting
        plt.figure(figsize=(12, 8))

        bar_width = 0.35
        index = range(len(resampled_data))

        plt.bar(index, resampled_data['Revenue_INR'], bar_width, label='Revenue (INR Crores)', color='green')
        plt.bar([i + bar_width for i in index], resampled_data['Earnings_INR'], bar_width, label='Earnings (INR Crores)', color='blue')

        if frequency_option == 'Quarterly':
            plt.xticks(index, [group.strftime('%Y-%m') for group in resampled_data.index])
        elif frequency_option == 'Annual':
            plt.xticks(index, [group.strftime('%Y') for group in resampled_data.index])

        plt.title('Historical Revenue and Earnings (INR Crores)')
        plt.xlabel('Date')
        plt.ylabel('Value (INR Crores)')
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        st.pyplot(plt)
    else:
        st.error("No historical stock data available for the specified time frame.")
else:
    st.error("End date should be greater than start date.")



# intrinsic  value

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate intrinsic value using DCF method
def calculate_intrinsic_value(stock_data, discount_rate, terminal_growth_rate):
    last_close_price = stock_data['Close'].iloc[-1]
    projected_cash_flows = [last_close_price * (1 + terminal_growth_rate) ** i for i in range(1, 6)]
    present_value = sum([cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(projected_cash_flows)])
    terminal_value = projected_cash_flows[-1] / (discount_rate - terminal_growth_rate)
    intrinsic_value = present_value + terminal_value / (1 + discount_rate) ** 5
    return intrinsic_value

# Streamlit app
st.title('Stock Intrinsic Value Calculator')

# User inputs
ticker = st.text_input('Enter Stock Ticker Symbol')
start_date = st.date_input('Enter Start Date')
end_date = st.date_input('Enter End Date')
discount_rate = st.number_input('Enter Discount Rate', value=0.1, step=0.01)
terminal_growth_rate = st.number_input('Enter Terminal Growth Rate', value=0.03, step=0.01)

# Fetch stock data
if ticker and start_date and end_date:
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if not stock_data.empty:
        st.subheader('Historical Stock Data')
        st.write(stock_data)

        # Calculate intrinsic value
        intrinsic_value = calculate_intrinsic_value(stock_data, discount_rate, terminal_growth_rate)
        st.subheader('Intrinsic Value')
        st.write(f'The intrinsic value of the share is: {intrinsic_value:.2f}')
    else:
        st.error('No historical stock data available for the specified time frame.')

