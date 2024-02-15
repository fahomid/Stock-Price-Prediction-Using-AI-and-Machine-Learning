import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle

def generate_data_for_date(date, scaler, historical_data, num_days=10):
    # Create a DataFrame with zeros and the specified date
    new_data = pd.DataFrame(index=[0], columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Date'])
    new_data['Date'] = date

    try:
        # Find the rows corresponding to the specified date in the historical data
        selected_data = historical_data[historical_data['Date'] == pd.to_datetime(date)]
        
        if not selected_data.empty:
            # Use the selected data as input features
            selected_data = selected_data.iloc[0]
            new_data[['Open', 'High', 'Low', 'Close', 'Volume']] = selected_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        else:
            raise ValueError("No data found for the specified date.")
    except ValueError as e:
        # error occurred becasue no date found in the csv for the date
        # selecting the last num_days rows before the specified date
        selected_data = historical_data.iloc[-num_days:]

        # calculating the average of the selected data
        average_data = selected_data.mean()

        # using the calculated average as input features
        new_data[['Open', 'High', 'Low', 'Close', 'Volume']] = average_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # scaling the features using the pre-trained scaler
    scaled_features = scaler.transform(new_data[['Open', 'High', 'Low', 'Close', 'Volume']])

    # assigning scaled features back to new_data
    new_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaled_features

    return new_data

# number of epochs
number_epochs = 500

# loading the pre-trained model
model = load_model(f'stock_price_prediction_model_{number_epochs}_epochs.keras')

# loading the pre-trained scaler
with open(f'scaler_{number_epochs}_epochs.pkl', 'rb') as f:
    scaler = pickle.load(f)

# loading historical data
historical_data = pd.read_csv('GOOG.csv')
historical_data['Date'] = pd.to_datetime(historical_data['Date'])

# ensuring data is sorted by date
historical_data = historical_data.sort_values(by='Date')

# specifying the date for which I want to generate predictions
specific_date = '2024-02-10'

# generating data for the specific date using the average of the last 10 days
input_data = generate_data_for_date(specific_date, scaler, historical_data, num_days=10)

# printing the generated input data
# print(f"Input Data for {specific_date}:")
# print(input_data)

# reshaping the input data for LSTM input and cast to float32
input_data_lstm = input_data[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape((1, 5, 1)).astype('float32')

# making predictions using the pre-trained model
predicted_close = model.predict(input_data_lstm)[0][0]

# printing the predicted close price
print(f'Predicted Close Price for {specific_date}: {predicted_close}')
