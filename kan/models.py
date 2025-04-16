import torch
import torch.nn as nn
from .layers import KANLayer, KANLayerEfficient


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network implementation.

    This implements the full KAN architecture as described in the paper:
    "Kolmogorov-Arnold Networks" (Liu et al., 2023)
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_sizes=None,
            grid_size=10,
            spline_type='cubic',
            efficient=True
    ):
        """
        Initialize a KAN model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_sizes (list): List of hidden layer sizes (default: None, just input->output)
            grid_size (int): Number of breakpoints in each spline
            spline_type (str): Type of spline to use ('linear' or 'cubic')
            efficient (bool): Whether to use the more efficient implementation
        """
        super().__init__()

        # Use efficient implementation if specified
        layer_class = KANLayerEfficient if efficient else KANLayer

        if hidden_sizes is None:
            # Create a single layer KAN (minimal model)
            self.layers = nn.ModuleList([
                layer_class(input_dim, output_dim, grid_size=grid_size, spline_type=spline_type)
            ])
        else:
            # Create a multi-layer KAN
            layer_sizes = [input_dim] + hidden_sizes + [output_dim]
            self.layers = nn.ModuleList()

            for i in range(len(layer_sizes) - 1):
                self.layers.append(
                    layer_class(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        grid_size=grid_size,
                        spline_type=spline_type
                    )
                )

    def forward(self, x):
        """
        Forward pass through the KAN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for comparison with KAN.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=None, activation=nn.ReLU()):
        """
        Initialize an MLP model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_sizes (list): List of hidden layer sizes
            activation (nn.Module): Activation function to use
        """
        super().__init__()

        if hidden_sizes is None:
            # Create a single layer MLP
            self.layers = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        else:
            # Create a multi-layer MLP
            layer_sizes = [input_dim] + hidden_sizes + [output_dim]
            layers = []

            for i in range(len(layer_sizes) - 2):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(activation)

            # Add the final layer
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim)
        """
        return self.layers(x)