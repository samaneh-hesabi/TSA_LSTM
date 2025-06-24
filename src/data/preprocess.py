import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    """
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of past values (X) and target values (y) for time series prediction.
    
    Example:
        If data = [1, 2, 3, 4, 5] and sequence_length = 2
        X will be [[1, 2], [2, 3], [3, 4]] (sequences of past values)
        y will be [3, 4, 5] (next value after each sequence)
    
    Args:
        data: Input time series data (should be normalized)
        sequence_length: Number of past time steps to use for prediction
        
    Returns:
        X: Sequences of past values
        y: Target values to predict
    """
    X_local, y_local = [], []  # Build sequences
    
    # Create sequences of past values and their corresponding target values
    for i in range(len(data) - sequence_length):
        # Past sequence: take 'sequence_length' consecutive values
        past_sequence = data[i:(i + sequence_length)]
        # Target: take the next value after the sequence
        target = data[i + sequence_length]
        
        X_local.append(past_sequence)
        y_local.append(target)
    
    return np.array(X_local), np.array(y_local)

def prepare_data(df: pd.DataFrame, 
                target_column: str,
                sequence_length: int = 10,
                train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare time series data for LSTM model training.
    
    Steps:
    1. Extract the target values we want to predict
    2. Scale the data to a range between 0 and 1
    3. Create sequences of past values for prediction
    4. Split the data into training and testing sets
    
    Args:
        df: Input DataFrame with time series data
        target_column: Name of the column we want to predict
        sequence_length: Number of past time steps to use for prediction
        train_size: Proportion of data to use for training (0.8 = 80%)
        
    Returns:
        X_train: Training sequences
        y_train: Training target values
        X_test: Testing sequences
        y_test: Testing target values
        scaler: The scaler used to normalize the data
    """
    # 1. Extract target values
    data = df[target_column].values.reshape(-1, 1)
    
    # 2. Scale the data between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 3. Create sequences
    X_received, y_received = prepare_sequences(scaled_data, sequence_length)
    
    # 4. Split into training and testing sets
    split_idx = int(len(X_received) * train_size)
    X_train, X_test = X_received[:split_idx], X_received[split_idx:]
    y_train, y_test = y_received[:split_idx], y_received[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    print("Data preprocessing module loaded successfully.")
