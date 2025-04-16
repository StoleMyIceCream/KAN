"""
Question 2: Apply your KAN to fit the function f(x, y) = sin(xy) + cos(x² + y²)
------------------------------------------------------------------------------

This script demonstrates fitting a KAN to the specified function and visualizes
the results including approximation quality and training progress.
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
from utils.metrics import mse, rmse, r2_score


def target_function(x, y):
    """The target function: f(x, y) = sin(xy) + cos(x² + y²)"""
    if isinstance(x, torch.Tensor):
        return torch.sin(x * y) + torch.cos(x ** 2 + y ** 2)
    else:
        return np.sin(x * y) + np.cos(x ** 2 + y ** 2)


def generate_data(n_samples=2000, x_range=(-3, 3), y_range=(-3, 3)):
    """Generate training data from the target function"""
    # Generate random points in the specified range
    X = torch.rand(n_samples, 2)
    X[:, 0] = X[:, 0] * (x_range[1] - x_range[0]) + x_range[0]  # Scale to x_range
    X[:, 1] = X[:, 1] * (y_range[1] - y_range[0]) + y_range[0]  # Scale to y_range

    # Compute function values
    y = target_function(X[:, 0], X[:, 1]).unsqueeze(1)

    return X, y


def create_test_grid(resolution=50, x_range=(-3, 3), y_range=(-3, 3)):
    """Create a regular grid for testing and visualization"""
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


def train_kan(n_samples=2000, epochs=2000, lr=0.005, hidden_sizes=None,
              grid_size=20, spline_type='cubic', batch_size=256):
    """Train a KAN to approximate the target function"""
    # Generate data
    X_train, y_train = generate_data(n_samples)

    # Create a test grid for evaluation
    X_test, y_test, X_grid, Y_grid = create_test_grid()

    # Create KAN model
    model = KAN(
        input_dim=2,
        output_dim=1,
        hidden_sizes=hidden_sizes,
        grid_size=grid_size,
        spline_type=spline_type,
        efficient=True
    )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    test_losses = []

    # Setup batching if needed
    if batch_size is None or batch_size >= n_samples:
        batch_size = n_samples  # Use all data
        n_batches = 1
    else:
        n_batches = n_samples // batch_size + (0 if n_samples % batch_size == 0 else 1)

    # Training loop with batching
    pbar = tqdm(range(epochs), desc="Training KAN")
    for epoch in pbar:
        # Shuffle data at each epoch
        indices = torch.randperm(n_samples)

        # Process mini-batches
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end_idx - start_idx)

        # Average loss for the epoch
        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)

        # Update progress bar
        if epoch % 10 == 0:
            pbar.set_postfix({
                "train_loss": f"{avg_loss:.6f}",
                "test_loss": f"{test_loss:.6f}"
            })

    # Final evaluation
    with torch.no_grad():
        test_outputs = model(X_test)

        # Reshape for visualization
        test_outputs_grid = test_outputs.reshape(X_grid.shape)
        y_test_grid = y_test.reshape(X_grid.shape)

        # Calculate metrics
        test_mse = mse(y_test, test_outputs)
        test_rmse = rmse(y_test, test_outputs)
        test_r2 = r2_score(y_test, test_outputs)

    print(f"Final test MSE: {test_mse:.6f}")
    print(f"Final test RMSE: {test_rmse:.6f}")
    print(f"Final test R²: {test_r2:.6f}")

    return model, losses, test_losses, test_outputs_grid, y_test_grid, X_grid, Y_grid


def visualize_results(model, losses, test_losses, pred_grid, true_grid, X_grid, Y_grid):
    """Visualize the KAN approximation and training progress"""
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: True function
    ax1 = fig.add_subplot(221, projection='3d')
    surf1 = ax1.plot_surface(X_grid.numpy(), Y_grid.numpy(), true_grid.squeeze().numpy(),
                             cmap='viridis', alpha=0.8)
    ax1.set_title("True Function: f(x, y) = sin(xy) + cos(x² + y²)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")

    # Plot 2: KAN approximation
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X_grid.numpy(), Y_grid.numpy(), pred_grid.squeeze().numpy(),
                             cmap='plasma', alpha=0.8)
    ax2.set_title("KAN Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x,y)")

    # Plot 3: Error surface
    ax3 = fig.add_subplot(223, projection='3d')
    error = torch.abs(pred_grid - true_grid)
    surf3 = ax3.plot_surface(X_grid.numpy(), Y_grid.numpy(), error.squeeze().numpy(),
                             cmap='hot', alpha=0.8)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("|error|")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    # Plot 4: Training and test losses
    ax4 = fig.add_subplot(224)
    ax4.plot(losses, label='Training Loss')
    ax4.plot(test_losses, label='Test Loss')
    ax4.set_title("Loss During Training")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("MSE Loss")
    ax4.set_yscale('log')  # Log scale for better visualization
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("q2_function_fitting_results.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train a KAN to fit a specific function")
    parser.add_argument("--samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden", type=int, nargs='+', default=None,
                        help="Hidden layer sizes (e.g., --hidden 10 10 for two hidden layers)")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size for splines")
    parser.add_argument("--spline-type", type=str, default="cubic", choices=["linear", "cubic"],
                        help="Type of spline to use")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    args = parser.parse_args()

    print("Training KAN to fit f(x, y) = sin(xy) + cos(x² + y²)...")
    model, losses, test_losses, pred_grid, true_grid, X_grid, Y_grid = train_kan(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        hidden_sizes=args.hidden,
        grid_size=args.grid_size,
        spline_type=args.spline_type,
        batch_size=args.batch_size
    )

    print("Visualizing results...")
    visualize_results(model, losses, test_losses, pred_grid, true_grid, X_grid, Y_grid)

    # Save model for later comparison
    torch.save(model.state_dict(), "kan_model_q2.pth")
    print("Model saved as 'kan_model_q2.pth'")

    print("Done!")


if __name__ == "__main__":
    main()