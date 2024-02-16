import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('Historical Revenue and Earnings')

# Get user input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Input for start date and end date
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Option selector for frequency
frequency_option = st.selectbox('Select Frequency', ['Quarterly', 'Annual'])

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
    resampled_data['Earnings_INR'] = resampled_data['Close'] * 0.1 * usd_to_inr_rate * inr_to_crores

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
