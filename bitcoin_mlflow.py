# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Importing the libraries and Data

import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.regularizers import l1
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reading the csv
btc_data = pd.read_csv("bitcoin_data.csv")

# Convert the index to column
btc_data = btc_data.reset_index()

#%% Sequences
sequence_length = 200  # Using the past 60 days to predict

# Define the function to create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length), 0]
        y = data[i + sequence_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

#%% MLFLOw
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Inputs 
dataset = mlflow.data.from_pandas(btc_data)
# Experiment
mlflow.set_experiment(experiment_name = "LSTM - Bitcoin")
#%% Modelo Padrão

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.001)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo 2x Epochs

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model Epochs 2x"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.001)
    mlflow.log_param("Epochs", 100)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo 2x Batch Size

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model Batch Size 2x"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.001)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 64)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo 2x Camadas

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model Layers 2x"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=True))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.001)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")

#%% Modelo 2x Unidades

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model Unity 2x"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(100, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.001)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo LR 0.01

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model LR 0.1"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo LR 0.0001

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model LR 0.0001"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.0001)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo Acitvation Function

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model ReLu"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25, activation='relu'))
    model_lstm.add(Dense(1, activation='linear'))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")

#%% Modelo Optimizer

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model RMSprop"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "RMSprop")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo Dropout

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model Dropout"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo L1

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model L1"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l1(0.01)))
    model_lstm.add(LSTM(50, return_sequences=False, kernel_regularizer=l1(0.01)))
    model_lstm.add(Dense(25, kernel_regularizer=l1(0.01)))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")
    
#%% Modelo L2

# Start the MLflow experiment
with mlflow.start_run(run_name="LSTM Bitcoin Model L2"):

    # ==================== INPUTS =============================================
    
    # Log the dataset and other relevant parameters
    mlflow.log_param("Dataset", "btc_data")
    mlflow.log_param("Sequence Length", sequence_length)

    # Normalize and process the data (same as before)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_close, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model (same as before)
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l1(0.01)))
    model_lstm.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)))
    model_lstm.add(Dense(25, kernel_regularizer=l2(0.01)))
    model_lstm.add(Dense(1))

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Log model architecture
    mlflow.log_param("Model Type", "LSTM")
    mlflow.log_param("Optimizer", "Adam")
    mlflow.log_param("Learning Rate", 0.01)
    mlflow.log_param("Epochs", 50)
    mlflow.log_param("Batch Size", 32)

    # Train the model
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # ==================== OUTPUTS ============================================

    # Evaluate on the test set
    test_loss = model_lstm.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model_lstm.predict(X_test)

    # Reverse the scaling transformation
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Log metrics to MLflow
    mlflow.log_metric("Test Loss", test_loss)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R²", r2)

    # Save and log the model as an artifact
    mlflow.keras.log_model(model_lstm, "lstm-bitcoin-model")

    # Plot training vs validation loss
    fig, ax = plt.subplots()
    ax.plot(history_lstm.history['loss'], label='Training Loss')
    ax.plot(history_lstm.history['val_loss'], label='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Log the plot as an artifact
    mlflow.log_figure(fig, "training_vs_validation_loss.png")

    # Save the trained model summary as a text file
    model_lstm_summary = []
    model_lstm.summary(print_fn=lambda x: model_lstm_summary.append(x))
    model_lstm_summary_text = "\n".join(model_lstm_summary)

    mlflow.log_text(model_lstm_summary_text, "lstm_model_summary.txt")