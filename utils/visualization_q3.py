from kan.models import KAN, MLP
import torch
import torch.nn as nn
from torchviz import make_dot
import argparse

def visualize_architectures(args):
    """
    Generates visualizations of KAN and MLP architectures using parameters
    from the command-line arguments.

    Args:
        args: An argparse.Namespace object containing the command-line arguments.
    """

    # KAN Model
    kan = KAN(
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

    mlp_deep = MLP(
        input_dim=2,
        output_dim=1,
        hidden_sizes=[args.hidden, args.hidden],
        activation=torch.nn.ReLU()
    )

    # Dummy input
    x = torch.randn(1, 2)  # Input dimension is always 2 in this script

    # Generate visualization for KAN
    kan_vis = make_dot(kan(x), params=dict(kan.named_parameters()))
    kan_vis.render("kan_shallow_visualization.png")

    # Generate visualization for MLP-Shallow
    mlp_shallow_vis = make_dot(mlp_shallow(x), params=dict(mlp_shallow.named_parameters()))
    mlp_shallow_vis.render("mlp_shallow_visualization.png")

    # Generate visualization for MLP-Deep
    mlp_deep_vis = make_dot(mlp_deep(x), params=dict(mlp_deep.named_parameters()))
    mlp_deep_vis.render("mlp_deep_visualization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare KAN against MLP and other models")
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

    visualize_architectures(args)