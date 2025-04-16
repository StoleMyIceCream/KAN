import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


def plot_2d_function(func, x_range=(-5, 5), y_range=(-5, 5), resolution=100, ax=None, title=None):
    """
    Plot a 2D function as a surface.

    Args:
        func (callable): Function to plot, should take vectors x, y and return z
        x_range (tuple): Range for x-axis (min, max)
        y_range (tuple): Range for y-axis (min, max)
        resolution (int): Resolution of the grid
        ax (matplotlib.axes.Axes): Axes to plot on, creates new figure if None
        title (str): Title for the plot

    Returns:
        matplotlib.figure.Figure: The figure object
        matplotlib.axes.Axes: The axes object
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    if hasattr(func, '__call__'):
        # Convert to tensor if needed
        if hasattr(func, 'parameters'):  # Check if this is a PyTorch model
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            inputs = torch.tensor(np.vstack((X_flat, Y_flat)).T, dtype=torch.float32)
            with torch.no_grad():
                Z_flat = func(inputs).numpy()
            Z = Z_flat.reshape(X.shape)
        else:
            # Regular function
            Z = func(X, Y)
    else:
        Z = func  # Assume it's already evaluated

    # Create plot
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')

    if title:
        ax.set_title(title)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    return fig, ax


def plot_loss_history(history, ax=None, title="Training Loss"):
    """
    Plot the loss history during training.

    Args:
        history (list): List of loss values
        ax (matplotlib.axes.Axes): Axes to plot on, creates new figure if None
        title (str): Title for the plot

    Returns:
        matplotlib.figure.Figure: The figure object
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(history)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True)

    return fig, ax


def plot_model_comparison(models, true_func, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
    """
    Compare multiple models' predictions against a true function.

    Args:
        models (dict): Dictionary of {model_name: model_function}
        true_func (callable): The true function to compare against
        x_range (tuple): Range for x-axis (min, max)
        y_range (tuple): Range for y-axis (min, max)
        resolution (int): Resolution of the grid

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Number of plots (true function + each model)
    n_plots = len(models) + 1

    # Create a grid of subplots
    fig = plt.figure(figsize=(15, 12))

    # Plot the true function
    ax1 = fig.add_subplot(2, (n_plots + 1) // 2, 1, projection='3d')
    plot_2d_function(true_func, x_range, y_range, resolution, ax=ax1, title='True Function')

    # Plot each model
    for i, (name, model) in enumerate(models.items(), start=2):
        ax = fig.add_subplot(2, (n_plots + 1) // 2, i, projection='3d')
        plot_2d_function(model, x_range, y_range, resolution, ax=ax, title=name)

    plt.tight_layout()
    return fig


def plot_loss_surface(model, param1_range, param2_range, loss_func, resolution=20):
    """
    Plot the loss surface by varying two parameters.

    Args:
        model: The model to analyze
        param1_range (tuple): (start, end, param_name, param_index)
        param2_range (tuple): (start, end, param_name, param_index)
        loss_func (callable): Function to compute loss given model
        resolution (int): Resolution of the grid

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract parameter information
    p1_start, p1_end, p1_name, p1_idx = param1_range
    p2_start, p2_end, p2_name, p2_idx = param2_range

    # Create parameter grids
    p1_values = np.linspace(p1_start, p1_end, resolution)
    p2_values = np.linspace(p2_start, p2_end, resolution)
    P1, P2 = np.meshgrid(p1_values, p2_values)

    # Store original parameter values
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute loss at each grid point
    loss_grid = np.zeros_like(P1)
    for i in range(resolution):
        for j in range(resolution):
            # Temporarily modify the parameters
            with torch.no_grad():
                model.state_dict()[p1_name][p1_idx] = torch.tensor(P1[i, j], dtype=torch.float32)
                model.state_dict()[p2_name][p2_idx] = torch.tensor(P2[i, j], dtype=torch.float32)

            # Compute loss
            loss_grid[i, j] = loss_func(model).item()

    # Restore original parameters
    model.load_state_dict(original_state_dict)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(P1, P2, loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    # Add labels
    ax.set_xlabel(f'{p1_name}[{p1_idx}]')
    ax.set_ylabel(f'{p2_name}[{p2_idx}]')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Surface')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    return fig, P1, P2, loss_grid