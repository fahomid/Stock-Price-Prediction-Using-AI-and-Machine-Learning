import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
import pickle

# loading the data from CSV file
file_path = 'GOOG.csv'
df = pd.read_csv(file_path)

# handling missing values
df = df.dropna()

# Select features (X) and target variable (y)
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# for this model I am targeting price closing column
target = 'Close'

X = df[features]
y = df[target]

# normalizing the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets, will use 20%, 80% split
X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(X_scaled, y, df['Date'], test_size=0.2, random_state=42)

# building the LSTM model
model = Sequential()
model.add(InputLayer(shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# reshaping the data for LSTM input
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# number of epochs
number_epochs = 5000

# training the model
model.fit(X_train_lstm, y_train, epochs=number_epochs, batch_size=32, verbose=2)

# saving the model in native Keras format
model.save(f'stock_price_prediction_model_{number_epochs}_epochs.keras')

# Save the scaler
with open(f'scaler_{number_epochs}_epochs.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully as scaler.pkl.")

print(f'\n\nModel saved successfully with {number_epochs} epochs!')