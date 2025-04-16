"""
Question 1: Build a minimal KAN
------------------------------

This script demonstrates a minimal implementation of a Kolmogorov-Arnold Network (KAN)
by training it to approximate a simple 2D function.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kan.models import KAN
from utils.visualization import plot_2d_function, plot_loss_history


def simple_function(x, y):
    """A simple 2D function for demonstration: f(x, y) = sin(x) + cos(y)"""
    if isinstance(x, torch.Tensor):
        return torch.sin(x) + torch.cos(y)
    else:
        return np.sin(x) + np.cos(y)


def generate_data(n_samples=1000, x_range=(-5, 5), y_range=(-5, 5)):
    """Generate training data from the simple function"""
    # Generate random points in the specified range
    X = torch.rand(n_samples, 2)
    X[:, 0] = X[:, 0] * (x_range[1] - x_range[0]) + x_range[0]  # Scale to x_range
    X[:, 1] = X[:, 1] * (y_range[1] - y_range[0]) + y_range[0]  # Scale to y_range

    # Compute function values
    y = simple_function(X[:, 0], X[:, 1]).unsqueeze(1)

    return X, y


def train_minimal_kan(n_samples=1000, epochs=1000, lr=0.01, grid_size=10, spline_type='cubic'):
    """Train a minimal KAN to approximate the simple function"""
    # Generate data
    X, y = generate_data(n_samples)

    # Create a minimal KAN with 2 inputs and 1 output
    model = KAN(
        input_dim=2,
        output_dim=1,
        hidden_sizes=None,  # No hidden layers (minimal KAN)
        grid_size=grid_size,
        spline_type=spline_type,
        efficient=True
    )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    pbar = tqdm(range(epochs), desc="Training KAN")
    for epoch in pbar:
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss
        losses.append(loss.item())

        # Update progress bar
        if epoch % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model, losses


def demonstrate_kan(model, losses):
    """Demonstrate the trained KAN by visualizing its predictions"""
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: True function
    ax1 = fig.add_subplot(131, projection='3d')
    plot_2d_function(simple_function, ax=ax1, title="True Function")

    # Plot 2: KAN approximation
    def model_function_wrapper(x, y):
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x.flatten()).float()
            y_tensor = torch.from_numpy(y.flatten()).float()
            inputs = torch.stack([x_tensor, y_tensor], dim=1)
            with torch.no_grad():
                outputs = model(inputs).squeeze().numpy()
            return outputs.reshape(x.shape)
        else:
            inputs = torch.stack([x, y], dim=1)
            with torch.no_grad():
                return model(inputs).squeeze()

    ax2 = fig.add_subplot(132, projection='3d')
    plot_2d_function(model_function_wrapper, ax=ax2, title="KAN Approximation")

    # Plot 3: Training loss
    ax3 = fig.add_subplot(133)
    plot_loss_history(losses, ax=ax3, title="Training Loss")

    plt.tight_layout()
    plt.savefig("q1_minimal_kan_results.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train a minimal KAN")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size for splines")
    parser.add_argument("--spline-type", type=str, default="cubic", choices=["linear", "cubic"],
                        help="Type of spline to use")
    args = parser.parse_args()

    print("Training a minimal KAN...")
    model, losses = train_minimal_kan(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        grid_size=args.grid_size,
        spline_type=args.spline_type
    )

    print("Visualizing results...")
    demonstrate_kan(model, losses)

    print("Done!")


if __name__ == "__main__":
    main()