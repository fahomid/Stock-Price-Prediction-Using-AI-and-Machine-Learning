import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# loading the data for prediction from CSV file
file_path = 'GOOG.csv'
df_new = pd.read_csv(file_path)

# handling missing values
df_new = df_new.dropna()

# selecting features (X) for prediction
X_new = df_new[features]

# normalizing the features using the same scaler from training
X_scaled_new = scaler.transform(X_new)

# reshaping the data for LSTM input
X_new_lstm = X_scaled_new.reshape((X_scaled_new.shape[0], X_scaled_new.shape[1], 1))

# setting epochs
number_epochs = 5000

# loading the saved model with number_epochs
loaded_model = load_model(f'stock_price_prediction_model_{number_epochs}_epochs.keras')

# making predictions
predictions = loaded_model.predict(X_new_lstm)

# creating a DataFrame with the predictions and corresponding dates
result_df = pd.DataFrame({
    'Date': df_new['Date'],
    'Actual': df_new['Close'],
    'Predicted': predictions.flatten()
})

# print the Date, Actual Price, and Predicted Price
for date, actual, predicted in zip(result_df['Date'], result_df['Actual'], result_df['Predicted']):
   print(f'Date: {date} | Actual Closing Price: {actual:.2f} | Predicted Closing Price: {predicted:.2f}')

# calculating the Mean Squared Error (MSE)
mse = ((result_df['Actual'] - result_df['Predicted']) ** 2).mean()

# calculating the Mean Absolute Error (MAE)
mae = abs(result_df['Actual'] - result_df['Predicted']).mean()

# printing the metrics
print(f'Model Trained with: {number_epochs} epoch(s)')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# saving actual price and predicted data into CSV file for later use
result_df.to_csv(f'prediction_results_with_{number_epochs}_epochs.csv', index=False)