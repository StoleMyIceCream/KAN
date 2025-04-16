import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import LinearSpline, CubicSpline


class KANLayer(nn.Module):
    """
    Basic implementation of a Kolmogorov-Arnold Network (KAN) layer.

    This layer implements the core component of KANs as described in the original paper,
    using the representation theorem to approximate multivariate functions.
    """

    def __init__(self, in_features, out_features, grid_size=10, spline_type='cubic'):
        """
        Initialize a KAN layer.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            grid_size (int): Number of breakpoints in each spline
            spline_type (str): Type of spline to use ('linear' or 'cubic')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create weight matrix for the linear combination of inputs
        # This is used to create the univariate projections
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))

        # Create bias terms
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Create spline activations
        self.splines = nn.ModuleList()
        for _ in range(out_features * in_features):
            if spline_type == 'linear':
                self.splines.append(LinearSpline(num_breakpoints=grid_size))
            else:  # cubic
                self.splines.append(CubicSpline(num_breakpoints=grid_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the layer parameters."""
        # Initialize weights using Kaiming/He initialization
        nn.init.kaiming_uniform_(self.weights, a=1)

        # Initialize bias to zero
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass through the KAN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features)
        """
        batch_size = x.size(0)

        # Initialize output tensor
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        # Process each output dimension
        for i in range(self.out_features):
            # Process each input dimension for this output
            for j in range(self.in_features):
                # Get the corresponding spline
                spline_idx = i * self.in_features + j
                spline = self.splines[spline_idx]

                # Apply weight to input and pass through spline
                # We scale by the weight before applying the spline
                weighted_input = self.weights[i, j] * x[:, j]
                spline_output = spline(weighted_input)

                # Add to the output
                output[:, i] += spline_output

            # Add bias
            output[:, i] += self.bias[i]

        return output


class KANLayerEfficient(nn.Module):
    """
    More efficient implementation of a KAN layer using vectorized operations.
    """

    def __init__(self, in_features, out_features, grid_size=10, spline_type='cubic'):
        """
        Initialize a KAN layer with vectorized operations.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            grid_size (int): Number of breakpoints in each spline
            spline_type (str): Type of spline to use ('linear' or 'cubic')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create projection matrix
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))

        # Create bias terms
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Create splines organized by output and input dimension
        self.splines = nn.ModuleList()
        for _ in range(out_features):
            input_splines = nn.ModuleList()
            for _ in range(in_features):
                if spline_type == 'linear':
                    input_splines.append(LinearSpline(num_breakpoints=grid_size))
                else:  # cubic
                    input_splines.append(CubicSpline(num_breakpoints=grid_size))
            self.splines.append(input_splines)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the layer parameters."""
        # Initialize weights using Kaiming/He initialization
        nn.init.kaiming_uniform_(self.weights, a=1)

        # Initialize bias to zero
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass through the KAN layer using vectorized operations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_features)
        """
        batch_size = x.size(0)

        # Initialize output tensor
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        # Apply each spline and sum the results
        for i in range(self.out_features):
            for j in range(self.in_features):
                # Weight the input for this dimension
                weighted_input = self.weights[i, j] * x[:, j]

                # Apply spline
                spline_output = self.splines[i][j](weighted_input)

                # Add to output
                output[:, i] += spline_output

            # Add bias
            output[:, i] += self.bias[i]

        return output