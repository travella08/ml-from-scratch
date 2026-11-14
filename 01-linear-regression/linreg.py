import numpy as np


def predict(X, w, b):
    """
    Predict y values with a linear model.

    Args:
        X: input features, shape (n, d)
        w: weights, one per feature, shape (d,)
        b: bias (intercept), a single number

    Returns:
        predictions, shape (n,)
    """
    return X @ w + b


def mse_loss(y_true, y_pred):
    """
    Mean squared error between true and predicted values.

    Args:
        y_true: actual values, shape (n,)
        y_pred: predicted values, shape (n,)

    Returns:
        a single number (the average squared error)
    """
    return np.mean((y_true - y_pred) ** 2)


def compute_gradients(X, y, w, b):
    """
    Gradients of MSE loss with respect to w and b.

    Args:
        X: input features, shape (n, d)
        y: true values, shape (n,)
        w: current weights, shape (d,)
        b: current bias, a single number

    Returns:
        (dw, db): dw has shape (d,), db is a single number
    """
    error = predict(X, w, b) - y
    dw = 2 * (X.T @ error) / len(y)
    db = 2 * np.mean(error)
    return dw, db


def train(X, y, lr=0.01, epochs=1000):
    """
    Train linear regression with gradient descent.

    Args:
        X: input features, shape (n, d)
        y: true values, shape (n,)
        lr: learning rate (step size)
        epochs: number of training steps

    Returns:
        (w, b, losses): final weights (shape (d,)), final bias,
        list of loss per epoch
    """
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []

    for _ in range(epochs):
        dw, db = compute_gradients(X, y, w, b)
        w -= dw * lr
        b -= db * lr
        losses.append(mse_loss(y, predict(X, w, b)))
    return w, b, losses
