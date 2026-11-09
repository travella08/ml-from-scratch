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

def compute_gradients(x, y, w, b):
    """
    Gradients of MSE loss with respect to w and b.

    Args:
        x: input values, shape (n,)
        y: true values, shape (n,)
        w: current weight
        b: current bias

    Returns:
        (dw, db): the two gradients, each a single number
    """
    error = predict(x, w, b) - y
    dw = 2 * np.mean(x * error)
    db = 2 * np.mean(error)
    return (dw, db)

def train(x, y, lr=0.01, epochs=1000):
    """
    Train linear regression with gradient descent.

    Args:
        x: input values, shape (n,)
        y: true values, shape (n,)
        lr: learning rate (step size)
        epochs: number of training steps

    Returns:
        (w, b, losses): final weight, final bias, list of loss per epoch
    """
    w, b = 0, 0
    losses = []

    for epoch in range(epochs):
        dw, db = compute_gradients(x, y, w, b)
        w -= dw * lr
        b -= db * lr
        losses.append(mse_loss(y, predict(x, w, b)))
    return w, b, losses