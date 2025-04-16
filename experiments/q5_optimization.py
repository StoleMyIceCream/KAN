"""
Question 5: Discuss the implications for optimization dynamics
-------------------------------------------------------------

This script analyzes the optimization dynamics of KANs, focusing on:
- Saddle points
- Flat minima
- Convergence behavior
- Comparison with traditional neural networks
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kan.models import KAN, MLP
from utils.visualization import plot_2d_function


def target_function(x, y):
    """The target function: f(x, y) = sin(xy) + cos(x² + y²)"""
    if isinstance(x, torch.Tensor):
        return torch.sin(x * y) + torch.cos(x ** 2 + y ** 2)
    else:
        return np.sin(x * y) + np.cos(x ** 2 + y ** 2)


def generate_data(n_samples=1000, x_range=(-3, 3), y_range=(-3, 3)):
    """Generate data from the target function"""
    # Generate random points in the specified range
    X = torch.rand(n_samples, 2)
    X[:, 0] = X[:, 0] * (x_range[1] - x_range[0]) + x_range[0]  # Scale to x_range
    X[:, 1] = X[:, 1] * (y_range[1] - y_range[0]) + y_range[0]  # Scale to y_range

    # Compute function values
    y = target_function(X[:, 0], X[:, 1]).unsqueeze(1)

    return X, y


def create_test_grid(resolution=50, x_range=(-3, 3), y_range=(-3, 3)):
    """Create a regular grid for testing"""
    # Create grid
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Flatten and stack
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    grid_points = torch.stack([X_flat, Y_flat], dim=1)

    # Compute target values
    Z = target_function(X_flat, Y_flat).unsqueeze(1)

    return grid_points, Z, X, Y


def compute_hessian_eigen(model, X, y, param_names=None):
    """
    Compute eigenvalues of the Hessian for specified parameters.

    Args:
        model: The model
        X, y: Training data
        param_names: List of parameter names to include (None for all)

    Returns:
        eigenvalues: List of eigenvalues of the Hessian
    """
    # Define loss function
    criterion = torch.nn.MSELoss()

    # Filter parameters
    if param_names is None:
        parameters = list(model.parameters())
    else:
        parameters = []
        for name, param in model.named_parameters():
            if any(p in name for p in param_names):
                parameters.append(param)

    # Compute loss
    outputs = model(X)
    loss = criterion(outputs, y)

    # Compute gradients
    grad_params = torch.autograd.grad(loss, parameters, create_graph=True)

    # Compute Hessian vector products for each parameter
    hessian_diag = []
    for param, grad in zip(parameters, grad_params):
        param_size = param.numel()

        # Only compute diagonal elements for large parameters
        if param_size > 100:
            # Flatten the gradient
            grad_flat = grad.reshape(-1)

            # Compute diagonal of Hessian (second derivatives)
            for i in tqdm(range(min(100, param_size)), desc=f"Computing Hessian for {param_size} params"):
                # Create a unit vector
                unit = torch.zeros_like(grad_flat)
                unit[i] = 1.0

                # Compute Hessian-vector product
                hv = torch.autograd.grad(grad_flat, param, unit, retain_graph=True)[0]
                hessian_diag.append(hv.reshape(-1)[i].item())
        else:
            # For small parameters, compute full Hessian
            hessian = torch.zeros(param_size, param_size)

            # Flatten the gradient
            grad_flat = grad.reshape(-1)

            # Compute full Hessian
            for i in tqdm(range(param_size), desc=f"Computing Hessian for {param_size} params"):
                # Create a unit vector
                unit = torch.zeros_like(grad_flat)
                unit[i] = 1.0

                # Compute Hessian-vector product
                hv = torch.autograd.grad(grad_flat, param, unit, retain_graph=True)[0]
                hessian[i] = hv.reshape(-1)

            # Compute eigenvalues
            try:
                eigenvalues = torch.linalg.eigvals(hessian).real
                hessian_diag.extend(eigenvalues.tolist())
            except Exception as e:
                print(f"Error computing eigenvalues: {e}")
                # Use diagonal as approximation
                hessian_diag.extend(torch.diag(hessian).tolist())

    return hessian_diag


def train_with_trajectory(model, X_train, y_train, epochs=500, lr=0.01, record_freq=10, record_hessian=False):
    """
    Train a model and record the trajectory of parameters and other metrics.

    Args:
        model: The model to train
        X_train, y_train: Training data
        epochs: Number of training epochs
        lr: Learning rate
        record_freq: Frequency of recording metrics
        record_hessian: Whether to compute and record Hessian eigenvalues

    Returns:
        dict: Dictionary of recorded metrics
    """
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize records
    records = {
        "loss": [],
        "grad_norm": [],
        "param_norms": [],
        "epochs": [],
        "time": [],
        "hessian_eigen": []
    }

    # Track important parameters for KANs
    param_trajectories = {}

    # Get key parameters to track
    key_params = []
    for name, param in model.named_parameters():
        if "weights" in name or "knots" in name or "values" in name:
            if param.numel() < 100:  # Only track small parameters to save memory
                key_params.append((name, param.clone()))
                param_trajectories[name] = []

    start_time = time.time()
    pbar = tqdm(range(epochs), desc=f"Training")

    for epoch in pbar:
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)

        # Update parameters
        optimizer.step()

        # Record metrics at specified frequency
        if epoch % record_freq == 0 or epoch == epochs - 1:
            # Record basic metrics
            records["loss"].append(loss.item())
            records["grad_norm"].append(grad_norm)
            records["epochs"].append(epoch)
            records["time"].append(time.time() - start_time)

            # Record parameter norms
            param_norm = 0.0
            for param in model.parameters():
                param_norm += param.norm().item() ** 2
            records["param_norms"].append(np.sqrt(param_norm))

            # Record key parameter values
            for name, _ in key_params:
                param_value = model.state_dict()[name].clone()
                if name not in param_trajectories:
                    param_trajectories[name] = []
                param_trajectories[name].append(param_value)

            # Compute Hessian eigenvalues (expensive)
            if record_hessian and (epoch % (record_freq * 10) == 0 or epoch == epochs - 1):
                print(f"\nComputing Hessian eigenvalues at epoch {epoch}...")
                hessian_eigen = compute_hessian_eigen(model, X_train, y_train)
                records["hessian_eigen"].append((epoch, hessian_eigen))

        # Update progress bar
        if epoch % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "grad_norm": f"{grad_norm:.6f}"
            })

    # Add parameter trajectories to records
    records["param_trajectories"] = param_trajectories

    return records


def plot_optimization_dynamics(kan_records, mlp_records):
    """
    Plot and compare optimization dynamics between KAN and MLP.

    Args:
        kan_records: Optimization records for KAN
        mlp_records: Optimization records for MLP
    """
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    # Plot 1: Loss curves
    axs[0, 0].plot(kan_records["epochs"], kan_records["loss"], label="KAN")
    axs[0, 0].plot(mlp_records["epochs"], mlp_records["loss"], label="MLP")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss (MSE)")
    axs[0, 0].set_title("Loss During Training")
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Gradient norm
    axs[0, 1].plot(kan_records["epochs"], kan_records["grad_norm"], label="KAN")
    axs[0, 1].plot(mlp_records["epochs"], mlp_records["grad_norm"], label="MLP")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Gradient Norm")
    axs[0, 1].set_title("Gradient Norm During Training")
    axs[0, 1].set_yscale('log')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Parameter norms
    axs[1, 0].plot(kan_records["epochs"], kan_records["param_norms"], label="KAN")
    axs[1, 0].plot(mlp_records["epochs"], mlp_records["param_norms"], label="MLP")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Parameter Norm")
    axs[1, 0].set_title("Parameter Norm During Training")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Training time
    axs[1, 1].plot(kan_records["epochs"], kan_records["time"], label="KAN")
    axs[1, 1].plot(mlp_records["epochs"], mlp_records["time"], label="MLP")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Cumulative Training Time (s)")
    axs[1, 1].set_title("Training Time")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("q5_optimization_dynamics.png", dpi=300)
    plt.show()

    # Plot Hessian eigenvalues if available
    if kan_records["hessian_eigen"] and mlp_records["hessian_eigen"]:
        plt.figure(figsize=(12, 8))

        # Get the last Hessian eigenvalues for both models
        kan_eigen = kan_records["hessian_eigen"][-1][1]
        mlp_eigen = mlp_records["hessian_eigen"][-1][1]

        # Plot histograms
        plt.hist(kan_eigen, bins=30, alpha=0.7, label="KAN")
        plt.hist(mlp_eigen, bins=30, alpha=0.7, label="MLP")

        plt.xlabel("Eigenvalue")
        plt.ylabel("Frequency")
        plt.title("Hessian Eigenvalue Distribution")
        plt.legend()
        plt.grid(True)

        plt.savefig("q5_hessian_eigenvalues.png", dpi=300)
        plt.show()

        # Also plot the sorted eigenvalues
        plt.figure(figsize=(12, 8))

        plt.semilogy(np.sort(np.abs(kan_eigen)), label="KAN")
        plt.semilogy(np.sort(np.abs(mlp_eigen)), label="MLP")

        plt.xlabel("Index")
        plt.ylabel("Absolute Eigenvalue (log scale)")
        plt.title("Sorted Hessian Eigenvalues")
        plt.legend()
        plt.grid(True)

        plt.savefig("q5_sorted_eigenvalues.png", dpi=300)
        plt.show()


def plot_parameter_trajectories(records, model_name):
    """
    Plot the trajectories of key parameters during training.

    Args:
        records: Training records with parameter trajectories
        model_name: Name of the model for the plot title
    """
    param_trajectories = records["param_trajectories"]

    # Select at most 4 parameters to plot
    selected_params = list(param_trajectories.keys())[:min(4, len(param_trajectories))]

    # Create a figure with subplots
    fig, axs = plt.subplots(len(selected_params), 1, figsize=(12, 4 * len(selected_params)))

    # If only one parameter, axs is not a list
    if len(selected_params) == 1:
        axs = [axs]

    # Plot each parameter trajectory
    for i, param_name in enumerate(selected_params):
        param_traj = param_trajectories[param_name]

        # Convert to numpy for plotting
        epochs = records["epochs"]

        # For each element in the trajectory, extract a slice to visualize
        # This is needed because parameters can be tensors of different shapes
        param_values = []

        # Check the shape of the parameter
        param_shape = param_traj[0].shape

        if len(param_shape) == 1:
            # If 1D, plot a few individual values
            indices = list(range(min(5, param_shape[0])))
            for j, idx in enumerate(indices):
                values = [p[idx].item() for p in param_traj]
                axs[i].plot(epochs, values, label=f"{param_name}[{idx}]")

        elif len(param_shape) == 2:
            # If 2D, plot a few individual values
            for j in range(min(2, param_shape[0])):
                for k in range(min(2, param_shape[1])):
                    values = [p[j, k].item() for p in param_traj]
                    axs[i].plot(epochs, values, label=f"{param_name}[{j},{k}]")

        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Parameter Value")
        axs[i].set_title(f"{param_name} Trajectory")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"q5_{model_name}_param_trajectories.png", dpi=300)
    plt.show()


def analyze_flatness(model, X, y, param_names=None, n_samples=100, perturb_scale=0.01):
    """
    Analyze the flatness of the loss landscape around the current model parameters.

    Args:
        model: The trained model
        X, y: Data for evaluating loss
        param_names: List of parameter names to perturb (None for all)
        n_samples: Number of perturbation samples
        perturb_scale: Scale of the perturbation

    Returns:
        means, stds: Mean and standard deviation of loss for each perturbation scale
    """
    # Define loss function
    criterion = torch.nn.MSELoss()

    # Get reference loss
    with torch.no_grad():
        outputs = model(X)
        ref_loss = criterion(outputs, y).item()

    print(f"Reference loss: {ref_loss:.6f}")

    # Filter parameters
    if param_names is None:
        parameters = list(model.named_parameters())
    else:
        parameters = [(name, param) for name, param in model.named_parameters()
                      if any(p in name for p in param_names)]

    # Store the original parameter values
    original_params = {name: param.clone() for name, param in parameters}

    # Try different scales of perturbation
    scales = np.logspace(-4, -1, 10)  # from 1e-4 to 1e-1

    means = []
    stds = []

    for scale in tqdm(scales, desc="Testing perturbation scales"):
        losses = []

        for _ in range(n_samples):
            # Perturb parameters
            with torch.no_grad():
                for name, param in parameters:
                    # Add Gaussian noise with scale relative to parameter norm
                    noise = torch.randn_like(param) * scale * param.norm() / param.numel() ** 0.5
                    param.copy_(original_params[name] + noise)

            # Compute loss with perturbed parameters
            with torch.no_grad():
                outputs = model(X)
                loss = criterion(outputs, y).item()

            losses.append(loss)

        # Restore original parameters
        with torch.no_grad():
            for name, param in parameters:
                param.copy_(original_params[name])

        # Compute statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        means.append(mean_loss)
        stds.append(std_loss)

        print(f"Scale {scale:.6f}: Mean loss = {mean_loss:.6f}, Std = {std_loss:.6f}")

    return scales, means, stds


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization dynamics of KANs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of data samples")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=50, help="Hidden layer size for models")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size for KAN splines")
    parser.add_argument("--record-freq", type=int, default=10,
                        help="Frequency of recording metrics")
    parser.add_argument("--record-hessian", action="store_true",
                        help="Whether to compute and record Hessian eigenvalues")

    args = parser.parse_args()

    # Generate data
    print("Generating data...")
    X_train, y_train = generate_data(n_samples=args.samples)
    X_test, y_test, X_grid, Y_grid = create_test_grid()

    # Create models
    print("Creating models...")

    # KAN Model
    kan_model = KAN(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden],
        grid_size=args.grid_size,
        spline_type='cubic',
        efficient=True
    )

    # MLP Model
    mlp_model = MLP(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden],
        activation=torch.nn.ReLU()
    )

    # Train KAN and record trajectory
    print("\nTraining KAN and recording trajectory...")
    kan_records = train_with_trajectory(
        kan_model, X_train, y_train,
        epochs=args.epochs,
        lr=args.lr,
        record_freq=args.record_freq,
        record_hessian=args.record_hessian
    )

    # Train MLP and record trajectory
    print("\nTraining MLP and recording trajectory...")
    mlp_records = train_with_trajectory(
        mlp_model, X_train, y_train,
        epochs=args.epochs,
        lr=args.lr,
        record_freq=args.record_freq,
        record_hessian=args.record_hessian
    )

    # Plot optimization dynamics
    print("\nPlotting optimization dynamics...")
    plot_optimization_dynamics(kan_records, mlp_records)

    # Plot parameter trajectories
    print("\nPlotting parameter trajectories...")
    plot_parameter_trajectories(kan_records, "kan")
    plot_parameter_trajectories(mlp_records, "mlp")

    # Analyze flatness of the loss landscape
    print("\nAnalyzing flatness of the loss landscape...")

    # For KAN
    print("KAN flatness:")
    kan_scales, kan_means, kan_stds = analyze_flatness(
        kan_model, X_test, y_test,
        param_names=["weights", "knots", "values"],
        n_samples=50,
        perturb_scale=0.01
    )

    # For MLP
    print("\nMLP flatness:")
    mlp_scales, mlp_means, mlp_stds = analyze_flatness(
        mlp_model, X_test, y_test,
        param_names=["weight", "bias"],
        n_samples=50,
        perturb_scale=0.01
    )

    # Plot flatness comparison
    plt.figure(figsize=(12, 8))

    plt.errorbar(kan_scales, kan_means, yerr=kan_stds, label="KAN", marker='o')
    plt.errorbar(mlp_scales, mlp_means, yerr=mlp_stds, label="MLP", marker='s')

    plt.xlabel("Perturbation Scale")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Flatness")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)

    plt.savefig("q5_flatness_comparison.png", dpi=300)
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()