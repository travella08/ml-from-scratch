import numpy as np


# ---------- model ----------

def sigmoid(z):
    """
    Squash any real number into (0, 1). Should be numerically stable for large |z|.

    Args:
        z: array of any shape

    Returns:
        array of the same shape, values in (0, 1)
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    for i in range(len(z)):
        if z[i] < 0:
            out[i] = np.exp(z[i]) / (1 + np.exp(z[i]))
        else:
            out[i] = 1 / (1 + np.exp(-z[i]))
    return out
def predict_proba(X, w, b):
    """
    Probability that each example belongs to class 1.

    Args:
        X: input features, shape (n, d)
        w: weights, shape (d,)
        b: bias, a single number

    Returns:
        probabilities, shape (n,)
    """
    return sigmoid(X @ w + b)

def predict(X, w, b, threshold=0.5):
    """
    Hard 0/1 class predictions.

    Args:
        X: input features, shape (n, d)
        w: weights, shape (d,)
        b: bias, a single number
        threshold: probability at or above which to predict class 1

    Returns:
        predicted labels (0 or 1), shape (n,)
    """
    probs = predict_proba(X, w, b)
    out = (probs >= threshold)
    return out.astype(int)



def bce_loss(y_true, y_prob):
    """
    Binary cross-entropy (log loss), averaged over examples. Guard against log(0).

    Args:
        y_true: true labels (0 or 1), shape (n,)
        y_prob: predicted probabilities, shape (n,)

    Returns:
        a single number
    """
    n = len(y_true)

    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    out = -1/n * sum(y_true * np.log(y_prob) + (1-y_true) * np.log(1-y_prob))

    return out


def compute_gradients(X, y_true, y_prob):
    """
    Gradients of the BCE loss with respect to w and b. Derive these on paper first.

    Args:
        X: input features, shape (n, d)
        y_true: true labels, shape (n,)
        y_prob: predicted probabilities, shape (n,)

    Returns:
        dw: shape (d,)
        db: a single number
    """
    raise NotImplementedError


def train(X, y, lr=0.1, epochs=1000):
    """
    Fit w and b with batch gradient descent.

    Args:
        X: input features, shape (n, d)
        y: true labels, shape (n,)
        lr: learning rate
        epochs: number of full passes over the data

    Returns:
        w: shape (d,)
        b: a single number
        losses: list of the training loss at each epoch, length epochs
    """
    raise NotImplementedError


# ---------- metrics ----------

def confusion_matrix(y_true, y_pred):
    """
    Args:
        y_true, y_pred: labels (0 or 1), shape (n,)

    Returns:
        2x2 array [[TN, FP], [FN, TP]] (same layout as scikit-learn)
    """
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """Fraction of predictions that are correct. Returns a single number."""
    raise NotImplementedError


def precision(y_true, y_pred):
    """TP / (TP + FP). Returns a single number; decide what to return when nothing is predicted positive."""
    raise NotImplementedError


def recall(y_true, y_pred):
    """TP / (TP + FN). Returns a single number."""
    raise NotImplementedError


def roc_curve(y_true, y_prob):
    """
    False positive rate and true positive rate as the threshold sweeps from high to low.

    Args:
        y_true: true labels, shape (n,)
        y_prob: predicted probabilities, shape (n,)

    Returns:
        fpr: shape (k,), starting at 0 and ending at 1
        tpr: shape (k,), starting at 0 and ending at 1
    """
    raise NotImplementedError


def auc(fpr, tpr):
    """Area under the ROC curve (trapezoid rule). Returns a single number in [0, 1]."""
    raise NotImplementedError
