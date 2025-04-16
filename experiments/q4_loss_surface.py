"""
Question 4: Analyze the loss surface of a shallow KAN with randomly initialized coefficients
-----------------------------------------------------------------------------------------

This script analyzes the loss surface of KANs by visualizing how the loss changes
when varying different parameters of the model.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import sys
import os
import itertools

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kan.models import KAN
from utils.visualization import plot_loss_surface


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


def compute_loss_surface_2d(model, X, y, param1_range, param2_range, resolution=20):
    """
    Compute the loss surface by varying two parameters.

    Args:
        model: The model whose loss surface to analyze
        X, y: Training data
        param1_range, param2_range: Tuples (min, max, param_name, param_index)
        resolution: Number of points in each dimension

    Returns:
        P1, P2: Meshgrids of parameter values
        loss_grid: Grid of loss values
    """
    # Extract parameter information
    p1_min, p1_max, p1_name, p1_idx = param1_range
    p2_min, p2_max, p2_name, p2_idx = param2_range

    # Create parameter grids
    p1_values = np.linspace(p1_min, p1_max, resolution)
    p2_values = np.linspace(p2_min, p2_max, resolution)
    P1, P2 = np.meshgrid(p1_values, p2_values)

    # Loss function
    criterion = torch.nn.MSELoss()

    # Store original parameter values
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute loss at each grid point
    loss_grid = np.zeros_like(P1)
    min_loss = float('inf')
    max_loss = float('-inf')

    # Progress bar
    total_iterations = resolution * resolution
    pbar = tqdm(total=total_iterations, desc="Computing loss surface")

    for i, j in itertools.product(range(resolution), range(resolution)):
        # Temporarily modify the parameters
        with torch.no_grad():
            try:
                model.state_dict()[p1_name][p1_idx] = torch.tensor(P1[i, j], dtype=torch.float32)
                model.state_dict()[p2_name][p2_idx] = torch.tensor(P2[i, j], dtype=torch.float32)
            except (IndexError, KeyError) as e:
                print(f"Error accessing {p1_name}[{p1_idx}] or {p2_name}[{p2_idx}]: {e}")
                continue

        # Compute loss
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            loss_grid[i, j] = loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        pbar.update(1)

    pbar.close()
    print(f"Loss range: Min = {min_loss:.6f}, Max = {max_loss:.6f}") # ADD THIS LINE

    # Restore original parameters
    model.load_state_dict(original_state_dict)

    return P1, P2, loss_grid

def analyze_loss_surface(X, y, hidden_sizes=None, grid_size=10, spline_type='cubic'):
    """
    Analyze the loss surface of a KAN with random initialization.

    Args:
        X, y: Training data
        hidden_sizes: Hidden layer sizes (None for shallow KAN)
        grid_size: Number of knots in splines
        spline_type: Type of spline ('linear' or 'cubic')
    """
    # Create a KAN with random initialization
    model = KAN(
        input_dim=2,
        output_dim=1,
        hidden_sizes=hidden_sizes,
        grid_size=grid_size,
        spline_type=spline_type,
        efficient=True
    )

    # Get model parameters for analysis
    param_dict = {name: param for name, param in model.named_parameters()}
    param_shapes = {name: param.shape for name, param in param_dict.items()}

    print("Model parameters:")
    for name, shape in param_shapes.items():
        print(f"  {name}: {shape}")

    # Find interesting parameters to analyze
    # For shallow KAN, we'll analyze:
    # 1. Weight parameters (how inputs are combined)
    # 2. Spline knot positions
    # 3. Spline values at knots

    # First, let's analyze varying weights
    print("\nAnalyzing weights...")
    weight_name = "layers.0.weights"

    if weight_name in param_dict:
        # Vary the first two weights and see effect on loss
        weight_range1 = (-5, 5, weight_name, (0, 0))  # First weight
        weight_range2 = (-5, 5, weight_name, (0, 1))  # Second weight

        print(f"Computing loss surface for weights ({weight_range1[3]} and {weight_range2[3]})...")
        P1, P2, loss_grid = compute_loss_surface_2d(
            model, X, y, weight_range1, weight_range2, resolution=30
        )

        # Plot 3D surface
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(P1, P2, loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax.set_xlabel(f'Weight 1 (input 1)')
        ax.set_ylabel(f'Weight 2 (input 2)')
        ax.set_zlabel('Loss')
        ax.set_title('Loss Surface: Varying Weights')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("q4_loss_surface_weights.png", dpi=300)
        plt.show()

        # Also create a contour plot for better visualization
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(P1, P2, loss_grid, 50, cmap='viridis')
        plt.colorbar(contour, label='Loss')
        plt.xlabel(f'Weight 1 (input 1)')
        plt.ylabel(f'Weight 2 (input 2)')
        plt.title('Loss Contour: Varying Weights')
        plt.savefig("q4_loss_contour_weights.png", dpi=300)
        plt.show()

    # Now analyze spline parameters
    print("\nAnalyzing spline parameters...")
    knots_name = "layers.0.splines.0.0.knots"  # First spline knots
    values_name = "layers.0.splines.0.0.values"  # First spline values

    if knots_name in param_dict and values_name in param_dict:
        # Vary the position and value of a specific knot
        knot_idx = grid_size // 2  # Middle knot
        knot_range = (-10, 10, knots_name, knot_idx)  # Wider range for knots
        value_range = (-5, 5, values_name, knot_idx)  # Wider range for values

        print(f"Computing loss surface for knot position and value (index {knot_idx})...")
        P1, P2, loss_grid = compute_loss_surface_2d(
            model, X, y, knot_range, value_range, resolution=30
        )

        # Plot 3D surface
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(P1, P2, loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax.set_xlabel(f'Knot Position')
        ax.set_ylabel(f'Knot Value')
        ax.set_zlabel('Loss')
        ax.set_title('Loss Surface: Varying Spline Parameters')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("q4_loss_surface_spline.png", dpi=300)
        plt.show()

        # Also create a contour plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(P1, P2, loss_grid, 50, cmap='viridis')
        plt.colorbar(contour, label='Loss')
        plt.xlabel(f'Knot Position')
        plt.ylabel(f'Knot Value')
        plt.title('Loss Contour: Varying Spline Parameters')
        plt.savefig("q4_loss_contour_spline.png", dpi=300)
        plt.show()

    # If KAN has derivatives (for cubic splines), analyze those too
    derivatives_name = "layers.0.splines.0.0.derivatives"
    if derivatives_name in param_dict and spline_type == 'cubic':
        # Vary the derivative at a knot and the knot position
        deriv_idx = grid_size // 2  # Middle knot
        deriv_range = (-5, 5, derivatives_name, deriv_idx)  # Wider range for derivatives
        knot_range = (-10, 10, knots_name, deriv_idx)  # Wider range for knots

        print(f"Computing loss surface for knot position and derivative (index {deriv_idx})...")
        P1, P2, loss_grid = compute_loss_surface_2d(
            model, X, y, knot_range, deriv_range, resolution=30
        )

        # Plot 3D surface
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(P1, P2, loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax.set_xlabel(f'Knot Position')
        ax.set_ylabel(f'Derivative Value')
        ax.set_zlabel('Loss')
        ax.set_title('Loss Surface: Varying Knot and Derivative')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("q4_loss_surface_derivative.png", dpi=300)
        plt.show()

        # Also create a contour plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(P1, P2, loss_grid, 50, cmap='viridis')
        plt.colorbar(contour, label='Loss')
        plt.xlabel(f'Knot Position')
        plt.ylabel(f'Derivative Value')
        plt.title('Loss Contour: Varying Knot and Derivative')
        plt.savefig("q4_loss_contour_derivative.png", dpi=300)
        plt.show()


def analyze_multiple_random_initializations(X, y, n_inits=5, hidden_sizes=None,
                                            grid_size=10, spline_type='cubic'):
    """
    Analyze the loss surfaces of multiple randomly initialized KANs.

    Args:
        X, y: Training data
        n_inits: Number of random initializations
        hidden_sizes: Hidden layer sizes (None for shallow KAN)
        grid_size: Number of knots in splines
        spline_type: Type of spline ('linear' or 'cubic')
    """
    print(f"\nAnalyzing {n_inits} random initializations...")

    # Parameters to analyze
    weight_name = "layers.0.weights"
    weight_range1 = (-5, 5, weight_name, (0, 0))  # First weight
    weight_range2 = (-5, 5, weight_name, (0, 1))  # Second weight

    # Store loss surfaces
    all_loss_grids = []

    for i in range(n_inits):
        print(f"\nInitialization {i + 1}/{n_inits}")

        # Create a KAN with random initialization
        model = KAN(
            input_dim=2,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            grid_size=grid_size,
            spline_type=spline_type,
            efficient=True
        )

        # Compute loss surface for this initialization
        print(f"Computing loss surface for weights...")
        P1, P2, loss_grid = compute_loss_surface_2d(
            model, X, y, weight_range1, weight_range2, resolution=20
        )

        all_loss_grids.append(loss_grid)

    # Plot average loss surface
    avg_loss_grid = np.mean(all_loss_grids, axis=0)

    # Plot 3D surface of average loss
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(P1, P2, avg_loss_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax.set_xlabel(f'Weight 1 (input 1)')
    ax.set_ylabel(f'Weight 2 (input 2)')
    ax.set_zlabel('Loss')
    ax.set_title('Average Loss Surface Across Initializations')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("q4_avg_loss_surface.png", dpi=300)
    plt.show()

    # Also create a contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(P1, P2, avg_loss_grid, 50, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    plt.xlabel(f'Weight 1 (input 1)')
    plt.ylabel(f'Weight 2 (input 2)')
    plt.title('Average Loss Contour Across Initializations')
    plt.savefig("q4_avg_loss_contour.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze the loss surface of KANs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of data samples")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Number of knots in splines")
    parser.add_argument("--spline-type", type=str, default="cubic", choices=["linear", "cubic"],
                        help="Type of spline to use")
    parser.add_argument("--hidden", type=int, nargs='+', default=None,
                        help="Hidden layer sizes (e.g., --hidden 10 for a single hidden layer)")
    parser.add_argument("--n-inits", type=int, default=5,
                        help="Number of random initializations to analyze")

    args = parser.parse_args()

    # Convert hidden argument to list or None
    hidden_sizes = args.hidden

    # Generate data
    print("Generating data...")
    X, y = generate_data(n_samples=args.samples)

    # Analyze loss surface of a single random initialization
    analyze_loss_surface(
        X, y,
        hidden_sizes=hidden_sizes,
        grid_size=args.grid_size,
        spline_type=args.spline_type
    )

    # Analyze multiple random initializations
    analyze_multiple_random_initializations(
        X, y,
        n_inits=args.n_inits,
        hidden_sizes=hidden_sizes,
        grid_size=args.grid_size,
        spline_type=args.spline_type
    )

    print("Done!")


if __name__ == "__main__":
    main()