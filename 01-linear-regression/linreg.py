import numpy as np

def predict(x, w, b):
    """
    Predict y values using a line.

    Args:
        x: input values, shape (n,)
        w: weight (slope), a single number
        b: bias (intercept), a single number

    Returns:
        predictions, shape (n,)
    """
    return w * x + b

def mse_loss(y_true, y_pred):
    """
    Mean squared error between true and predicted values.

    Args:
        y_true: actual values, shape (n,)
        y_pred: predicted values, shape (n,)

    Returns:
        a single number (the average squared error)
    """
    return np.mean((y_true-y_pred)**2)
