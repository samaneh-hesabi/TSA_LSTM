import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def create_model(sequence_length=5):
    """
    Create a simple LSTM model for time series prediction.
    
    Args:
        sequence_length: Number of past days to look at (default is 5)
                       For example, use last 5 days to predict next day
    
    Example:
        If we have stock prices [100, 101, 102, 103, 104], and sequence_length=3,
        we use [100, 101, 102] to predict 103
    """
    model = Sequential([
        # First layer: LSTM to process the sequence of past values
        LSTM(20, input_shape=(sequence_length, 1)),
        # 20 is the number of neurons in the LSTM layer: Think of it like having 20 different "memory cells" to remember different patterns
        # input_shape is the shape of the input data
        # (sequence_length, 1) means we're using the last 5 days to predict the next day
        
        # Second layer: Make the final prediction (one number)
        Dense(1)
    ])
    
    # Set up the model to minimize prediction errors
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, past_values, target_values, epochs=30):
    """
    Train the model to make predictions.
    
    Args:
        model: The model created by create_model()
        past_values: Past values to learn from
        target_values: The actual next values we want to predict
        epochs: How many times to practice on the data (default 30)
    """
    history = model.fit(
        past_values, target_values,
        epochs=epochs,
        batch_size=32,  # Look at 32 examples at once
        validation_split=0.2,  # Use 20% of data to check progress
        verbose=1  # Show training progress
    )
    return history

def make_prediction(model, past_values):
    """
    Use the trained model to predict next values.
    
    Args:
        model: The trained model
        past_values: Past values to use for prediction
    
    Returns:
        Predicted next values
    """
    return model.predict(past_values, verbose=0)

def check_accuracy(model, test_data, true_values):
    """
    Check how well the model predicts.
    
    Args:
        model: The trained model
        test_data: Past values to test prediction on
        true_values: The actual values that should have been predicted
    
    Returns:
        Average prediction error (lower is better)
    """
    return model.evaluate(test_data, true_values, verbose=0)

# Example of how to use these functions:
if __name__ == "__main__":
    # This is just an example - you'll need your own data
    print("Here's how to use this model:")
    print("1. Create the model:")
    print("   model = create_model(sequence_length=5)")
    print("2. Train it:")
    print("   train_model(model, your_past_values, your_target_values)")
    print("3. Make predictions:")
    print("   predictions = make_prediction(model, new_past_values)") 