import numpy as np

def split_data(X, y, train_size=0.8):
    """Split the data into training and testing datasets."""
    idx = int(len(X) * train_size)
    X_train, X_test = X[:idx], X[idx:]
    y_train, y_test = y[:idx], y[idx:]
    return X_train, X_test, y_train, y_test
