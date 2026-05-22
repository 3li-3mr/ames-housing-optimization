import pandas as pd
import numpy as np
import os

def load_and_prep_data(filepath='data/AmesHousing.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find dataset at {filepath}. Please ensure it is downloaded and placed in the 'data' folder.")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Extract independent (X) and dependent (y) variables
    X_raw = df['Gr Liv Area'].values
    y_raw = df['SalePrice'].values
    
    # Normalize using Z-score standardization
    # Formula: (value - mean) / standard_deviation
    X_normalized = (X_raw - np.mean(X_raw)) / np.std(X_raw)
    y_normalized = (y_raw - np.mean(y_raw)) / np.std(y_raw)
    
    return X_normalized, y_normalized

# the model
def predict(X, a0, a1):
    return a0 + a1 * X

# loss fn
def calculate_sse_loss(X, y, a0, a1):
    y_pred = predict(X, a0, a1)
    return 0.5 * np.sum((y - y_pred)**2)


def compute_gradients(X, y, a0, a1):

    y_pred = predict(X, a0, a1)
    error = y - y_pred
    
    # The negative sign comes from the chain rule derivative of (y - y_pred)^2
    grad_a0 = -np.sum(error)
    grad_a1 = -np.sum(error * X)
    
    return np.array([grad_a0, grad_a1])


def compute_hessian(X):

    N = len(X)
    sum_x = np.sum(X)
    sum_x2 = np.sum(X**2)
    
    # Hessian matrix construction
    H = np.array([
        [N,      sum_x],
        [sum_x,  sum_x2]
    ])
    
    return H