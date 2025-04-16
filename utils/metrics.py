import numpy as np
import torch


def mse(y_true, y_pred):
    """
    Mean Squared Error between true values and predictions.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: Mean squared error
    """
    if isinstance(y_true, torch.Tensor):
        return torch.mean((y_true - y_pred) ** 2).item()
    else:
        return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error between true values and predictions.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: Root mean squared error
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error between true values and predictions.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: Mean absolute error
    """
    if isinstance(y_true, torch.Tensor):
        return torch.mean(torch.abs(y_true - y_pred)).item()
    else:
        return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    R² score (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: R² score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Handle edge case where ss_tot is 0
    if ss_tot == 0:
        return 0  # If data has no variance, R² is undefined

    return 1 - (ss_res / ss_tot)


def convergence_rate(loss_history, window_size=10):
    """
    Calculate convergence rates from loss history.

    Args:
        loss_history (list): List of loss values during training
        window_size (int): Size of window for calculating convergence rate

    Returns:
        list: Convergence rates (percentage change in loss per window)
    """
    rates = []

    for i in range(window_size, len(loss_history), window_size):
        prev_window = loss_history[i - window_size:i - window_size // 2]
        curr_window = loss_history[i - window_size // 2:i]

        prev_avg = sum(prev_window) / len(prev_window)
        curr_avg = sum(curr_window) / len(curr_window)

        if prev_avg == 0:
            rate = 0  # Avoid division by zero
        else:
            rate = (prev_avg - curr_avg) / prev_avg

        rates.append(rate)

    return rates