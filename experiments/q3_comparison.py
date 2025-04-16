"""
Question 3: Compare the convergence and fit quality against a shallow MLP
-------------------------------------------------------------------------

This script compares the KAN implementation against a shallow MLP and potentially
other neural network architectures, analyzing both convergence behavior and
fit quality.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kan.models import KAN, MLP
from utils.visualization import plot_2d_function, plot_loss_history, plot_model_comparison
from utils.metrics import mse, rmse, r2_score, convergence_rate


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


def train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=1000, lr=0.01, batch_size=256):
    """Train a model and track performance metrics"""
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Calculate number of batches
    n_samples = len(X_train)
    if batch_size is None or batch_size >= n_samples:
        batch_size = n_samples  # Use all data
        n_batches = 1
    else:
        n_batches = n_samples // batch_size + (0 if n_samples % batch_size == 0 else 1)

    # Training loop
    train_losses = []
    val_losses = []
    test_losses = []
    training_times = []

    # Track best model
    best_val_loss = float('inf')
    best_model_state = None

    start_time = time.time()
    pbar = tqdm(range(epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        epoch_start = time.time()

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
        train_losses.append(avg_loss)

        # Evaluate on validation set
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

            # Evaluate on test set
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Record time per epoch
        epoch_time = time.time() - epoch_start
        training_times.append(epoch_time)

        # Update progress bar
        if epoch % 10 == 0:
            pbar.set_postfix({
                "train_loss": f"{avg_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "test_loss": f"{test_loss:.6f}",
                "time/epoch": f"{epoch_time:.3f}s"
            })

    total_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    with torch.no_grad():
        test_outputs = model(X_test)
        final_test_loss = criterion(test_outputs, y_test).item()
        final_test_rmse = rmse(y_test, test_outputs)
        final_test_r2 = r2_score(y_test, test_outputs)

    # Calculate convergence rate
    conv_rates = convergence_rate(train_losses, window_size=max(10, epochs // 20))

    # Return results
    results = {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "training_times": training_times,
        "total_time": total_time,
        "final_test_loss": final_test_loss,
        "final_test_rmse": final_test_rmse,
        "final_test_r2": final_test_r2,
        "convergence_rates": conv_rates,
    }

    return results


def evaluate_models(models_results, X_grid, y_grid, X_grid_2d, Y_grid_2d):
    """Evaluate and compare all trained models"""
    # Create predictions for each model
    model_predictions = {}

    for name, results in models_results.items():
        model = results["model"]
        with torch.no_grad():
            preds = model(X_grid)
            preds_grid = preds.reshape(X_grid_2d.shape)
            model_predictions[name] = preds_grid

    # Create error grids
    model_errors = {}
    for name, preds_grid in model_predictions.items():
        error_grid = torch.abs(preds_grid - y_grid.reshape(X_grid_2d.shape))
        model_errors[name] = error_grid

    # Visualize predictions
    fig = plt.figure(figsize=(18, 12))

    # True function
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X_grid_2d.numpy(), Y_grid_2d.numpy(), y_grid.reshape(X_grid_2d.shape).squeeze().numpy(),
                             cmap='viridis', alpha=0.8)
    ax1.set_title("True Function")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")

    # Position in the grid for each model
    positions = {
        list(models_results.keys())[0]: 232,
        list(models_results.keys())[1]: 233,
        list(models_results.keys())[2] if len(models_results) > 2 else None: 234,
    }

    # Model predictions
    for name, pos in positions.items():
        if name is None:
            continue

        ax = fig.add_subplot(pos, projection='3d')
        surf = ax.plot_surface(X_grid_2d.numpy(), Y_grid_2d.numpy(), model_predictions[name].squeeze().numpy(),
                               cmap='plasma', alpha=0.8)
        ax.set_title(f"{name} Prediction")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")

    # Error comparison
    ax4 = fig.add_subplot(235)
    for name, losses in models_results.items():
        ax4.plot(losses["train_losses"], label=f"{name} Train")
        ax4.plot(losses["val_losses"], linestyle='--', label=f"{name} Val")

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (MSE)")
    ax4.set_title("Loss Curves")
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)

    # Convergence rate comparison
    ax5 = fig.add_subplot(236)
    for name, results in models_results.items():
        rates = results["convergence_rates"]
        epochs = np.linspace(0, len(results["train_losses"]), len(rates))
        ax5.plot(epochs, rates, marker='o', linestyle='-', label=name)

    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Convergence Rate")
    ax5.set_title("Convergence Comparison")
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig("q3_model_comparison.png", dpi=300)
    plt.show()

    # Show error metrics table
    print("\nError Metrics Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'MSE':<12} {'RMSE':<12} {'R²':<12} {'Training Time':<15}")
    print("-" * 60)

    for name, results in models_results.items():
        print(f"{name:<15} {results['final_test_loss']:<12.6f} {results['final_test_rmse']:<12.6f} "
              f"{results['final_test_r2']:<12.6f} {results['total_time']:<15.2f}s")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare KAN against MLP and other models")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=500,
                        help="Number of validation samples")
    parser.add_argument("--test-samples", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--hidden", type=int, default=50,
                        help="Hidden layer size for shallow networks")
    args = parser.parse_args()

    # Generate datasets
    print("Generating datasets...")
    X_train, y_train = generate_data(n_samples=args.samples)
    X_val, y_val = generate_data(n_samples=args.val_samples)
    X_test, y_test = generate_data(n_samples=args.test_samples)

    # Create a grid for visualization
    X_grid, y_grid, X_grid_2d, Y_grid_2d = create_test_grid()

    # Create models
    print("Creating models...")

    # KAN Models
    kan_shallow = KAN(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden],
        grid_size=20,
        spline_type='cubic',
        efficient=True
    )

    # MLP Models
    mlp_shallow = MLP(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden],
        activation=torch.nn.ReLU()
    )

    # Deeper MLP with same number of parameters as KAN
    mlp_deep = MLP(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden, args.hidden],
        activation=torch.nn.ReLU()
    )

    # Train models
    print("Training models...")
    models_results = {}

    # Train KAN
    print("\nTraining Shallow KAN:")
    kan_results = train_model(
        kan_shallow, "KAN",
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    models_results["KAN"] = kan_results

    # Train Shallow MLP
    print("\nTraining Shallow MLP:")
    mlp_shallow_results = train_model(
        mlp_shallow, "MLP-Shallow",
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    models_results["MLP-Shallow"] = mlp_shallow_results

    # Train Deep MLP
    print("\nTraining Deep MLP:")
    mlp_deep_results = train_model(
        mlp_deep, "MLP-Deep",
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    models_results["MLP-Deep"] = mlp_deep_results

    # Evaluate and compare all models
    print("\nEvaluating models...")
    evaluate_models(models_results, X_grid, y_grid, X_grid_2d, Y_grid_2d)

    # Save models
    print("\nSaving models...")
    os.makedirs("../models", exist_ok=True)
    torch.save(kan_shallow.state_dict(), "../models/kan_q3.pth")
    torch.save(mlp_shallow.state_dict(), "../models/mlp_shallow_q3.pth")
    torch.save(mlp_deep.state_dict(), "../models/mlp_deep_q3.pth")

    print("Done!")


if __name__ == "__main__":
    main()