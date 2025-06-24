<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Source Code Structure</div>

This directory contains the main source code for the time series prediction project.

# 1. Directory Structure

```
src/
├── data/
│   └── preprocess.py     # Data preparation functions
└── models/
    ├── lstm_model.py     # Simple LSTM model implementation
    └── train.py          # Training script
```

# 2. Components

## 2.1 Data Processing (`data/preprocess.py`)
- Loads time series data from CSV files
- Scales data between 0 and 1
- Creates sequences for LSTM training
- Splits data into training and testing sets

## 2.2 LSTM Model (`models/lstm_model.py`)
- Simple LSTM model implementation
- Single LSTM layer for sequence processing
- Dense layer for making predictions

## 2.3 Training Script (`models/train.py`)
- Main script to train the model
- Loads and prepares data
- Trains the model
- Visualizes predictions 