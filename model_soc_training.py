import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
import os

def load_and_preprocess_data():
    df = pd.read_csv('data/battery_test_data.csv')
    
    # Use these features for prediction
    X = df[['voltage', 'current', 'temperature', 'internal_resistance']].values
    y = df[['soc_actual', 'soh_actual']].values
    
    # Normalize data
    x_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)
    
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y)  # Scale both SOC and SOH to [0,1]
    
    # Create sequences
    def create_sequences(X, y, seq_length=10):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = create_sequences(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2)
    
    return X_train, X_test, y_train, y_test, x_scaler, y_scaler

def build_model(sequence_length, n_features):
    # Shared LSTM layers
    inputs = Input(shape=(sequence_length, n_features))
    lstm1 = LSTM(64, return_sequences=True)(inputs)
    lstm2 = LSTM(32)(lstm1)
    
    # Separate output branches
    soc_output = Dense(1, activation='sigmoid', name='soc')(lstm2)
    soh_output = Dense(1, activation='sigmoid', name='soh')(lstm2)
    
    model = Model(inputs=inputs, outputs=[soc_output, soh_output])
    
    model.compile(
        optimizer='adam',
        loss={'soc': 'mse', 'soh': 'mse'},
        metrics={'soc': 'mae', 'soh': 'mae'},
        loss_weights={'soc': 0.7, 'soh': 0.3}  # SOC is more important
    )
    
    return model

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = load_and_preprocess_data()
    
    # Build and train model
    model = build_model(X_train.shape[1], X_train.shape[2])
    model.fit(
        X_train,
        {'soc': y_train[:, 0], 'soh': y_train[:, 1]},
        epochs=50,
        batch_size=32,
        validation_data=(X_test, {'soc': y_test[:, 0], 'soh': y_test[:, 1]})
    )
    
    # Save model and scalers
    os.makedirs('models', exist_ok=True)
    model.save('models/battery_lstm.keras')
    
    import joblib
    joblib.dump(x_scaler, 'models/x_scaler.save')
    joblib.dump(y_scaler, 'models/y_scaler.save')
    
    print("Model and scalers saved in 'models' directory")