import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os # To save the plot

print("TensorFlow Version:", tf.__version__)

# --- 1. Generate Synthetic Time Series Data (Sine Wave with Noise) ---
print("Generating synthetic data...")
time = np.arange(0, 500, 0.1)
# Introduce amplitude variation and noise for slight complexity
amplitude = np.sin(time / 10) + np.sin(time / 5) # Varying amplitude
noise = np.random.normal(scale=0.1, size=time.shape)
data = amplitude * np.sin(time) + noise
# data = np.sin(time) + np.random.normal(scale=0.1, size=time.shape) # Simpler sine wave

# Create a DataFrame for easier handling (optional but good practice)
df = pd.DataFrame({'value': data})

plt.figure(figsize=(12, 4))
plt.title("Generated Sine Wave Data with Noise")
plt.plot(time, df['value'])
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
# Save the plot of raw data
plot_filename_raw = "generated_data_plot.png"
plt.savefig(plot_filename_raw)
print(f"Raw data plot saved to {plot_filename_raw}")
# plt.show() # Don't show plots if running non-interactively

# --- 2. Preprocessing ---
print("Preprocessing data...")
# Scale data to be between 0 and 1 (common for LSTMs)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['value'].values.reshape(-1, 1))

# Create sequences (input: sequence, output: next value)
def create_sequences(data, look_back=50):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

LOOK_BACK_PERIOD = 50 # How many previous time steps to use for prediction
X, y = create_sequences(scaled_data, LOOK_BACK_PERIOD)

# Reshape input to be [samples, time steps, features] (required by LSTM)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(f"Shape of input sequences (X): {X.shape}") # Should be (samples, LOOK_BACK_PERIOD, 1)
print(f"Shape of target values (y): {y.shape}")   # Should be (samples,)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Shuffle=True by default
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --- 3. Build the LSTM Model ---
print("Building LSTM model...")
model = Sequential()

# Add LSTM layers
# Use return_sequences=True if stacking LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK_PERIOD, 1)))
model.add(Dropout(0.2)) # Dropout for regularization

model.add(LSTM(units=50, return_sequences=False)) # Last LSTM layer returns only the final output
model.add(Dropout(0.2))

# Add a Dense output layer (1 neuron for predicting the single next value)
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error') # Adam optimizer and MSE loss are common for regression

model.summary() # Print model architecture

# --- 4. Train the Model ---
print("Training model...")
# Early stopping to prevent overfitting and save time
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,           # Increase epochs if needed, but start low
    batch_size=32,       # Adjust batch size based on memory
    validation_split=0.1, # Use part of training data for validation during training
    callbacks=[early_stopping],
    verbose=1            # Show training progress
)

# --- 5. Evaluate the Model ---
print("Evaluating model...")
# Make predictions on the test set
predictions_scaled = model.predict(X_test)

# Inverse transform predictions and actual values to original scale
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test_actual)**2))
print(f"Test Set Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- 6. Visualize Results ---
print("Visualizing results...")

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plot_filename_loss = "training_loss_plot.png"
plt.savefig(plot_filename_loss)
print(f"Loss plot saved to {plot_filename_loss}")
# plt.show()

# Plot Actual vs Predicted values on the Test Set
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Values', color='blue', alpha=0.7)
plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')
plt.title('Actual vs Predicted Values on Test Set')
plt.xlabel('Time Steps (Test Sample Index)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plot_filename_pred = "prediction_plot.png"
plt.savefig(plot_filename_pred)
print(f"Prediction plot saved to {plot_filename_pred}")
# plt.show()

print("Script finished successfully!")