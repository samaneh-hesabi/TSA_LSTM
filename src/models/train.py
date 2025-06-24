import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import tensorflow as tf

# Suppress TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import load_data, prepare_data
from models.lstm_model import create_model, train_model, make_prediction

def train_and_evaluate(data_path: str, target_column: str = "price", sequence_length: int = 10):
    """
    Simple function to train and evaluate LSTM model on time series data.
    """
    # Load and prepare data
    df = load_data(data_path)
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df=df,
        target_column=target_column,
        sequence_length=sequence_length
    )
    
    # Create and train model
    model = create_model(sequence_length=sequence_length)
    train_model(model, X_train, y_train, epochs=30)
    
    # Make predictions
    test_pred = make_prediction(model, X_test)
    
    # Convert predictions back to original scale
    test_pred = scaler.inverse_transform(test_pred)
    y_test_orig = scaler.inverse_transform(y_test)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_orig, label='Actual', color='blue')
    plt.plot(test_pred, label='Predicted', color='red', linestyle='--')
    plt.title('Test Set: Predictions vs Actual Values')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data", "sp500_data.csv")
    train_and_evaluate(data_path) 