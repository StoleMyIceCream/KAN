import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSpline(nn.Module):
    """
    Linear spline activation function for KAN.

    This implements a continuous piecewise linear function with learnable knots and slopes.
    """

    def __init__(self, num_breakpoints=10, init_range=(-3, 3), zero_init=True):
        """
        Initialize a linear spline function.

        Args:
            num_breakpoints (int): Number of breakpoints in the spline
            init_range (tuple): Initial range for the breakpoints (min, max)
            zero_init (bool): If True, initialize the output weights to 0
        """
        super().__init__()

        # Create evenly spaced knots in the given range
        self.num_breakpoints = num_breakpoints
        knots = torch.linspace(init_range[0], init_range[1], num_breakpoints)

        # Initialize parameters
        self.knots = nn.Parameter(knots)

        # Initialize the values at each knot
        if zero_init:
            self.values = nn.Parameter(torch.zeros(num_breakpoints))
        else:
            self.values = nn.Parameter(torch.randn(num_breakpoints) * 0.01)

    def forward(self, x):
        """
        Evaluate the spline function at points x.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output of the spline function
        """
        # Expand knots for batch processing
        # shape: (num_breakpoints, 1)
        knots = self.knots.view(-1, 1)

        # Expand values for batch processing
        # shape: (num_breakpoints, 1)
        values = self.values.view(-1, 1)

        # Compute distances between input and knots
        # Expand x to shape (1, batch_size)
        x_expanded = x.view(1, -1)

        # Calculate distances: |x - knot_i|
        # shape: (num_breakpoints, batch_size)
        distances = torch.abs(x_expanded - knots)

        # For each input, find the closest knot
        # shape: (batch_size,)
        _, indices = torch.min(distances, dim=0)

        # Extract the values at the closest knot for each input
        # shape: (batch_size,)
        result = torch.gather(values.view(-1), 0, indices)

        return result


class CubicSpline(nn.Module):
    """
    Cubic spline activation function for KAN.

    This implements a continuous piecewise cubic function with learnable knots and parameters.
    It ensures C^2 continuity (continuous second derivatives) at the knots.
    """

    def __init__(self, num_breakpoints=10, init_range=(-3, 3)):
        """
        Initialize a cubic spline function.

        Args:
            num_breakpoints (int): Number of breakpoints in the spline
            init_range (tuple): Initial range for the breakpoints (min, max)
        """
        super().__init__()

        # Create evenly spaced knots in the given range
        self.num_breakpoints = num_breakpoints
        knots = torch.linspace(init_range[0], init_range[1], num_breakpoints)

        # Initialize knots as parameters
        self.knots = nn.Parameter(knots)

        # Initialize the values at each knot
        self.values = nn.Parameter(torch.zeros(num_breakpoints))

        # Initialize derivatives at each knot
        # For cubic spline, we need 1st derivatives at each knot
        self.derivatives = nn.Parameter(torch.zeros(num_breakpoints))

    def forward(self, x):
        """
        Evaluate the cubic spline function at points x.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output of the spline function
        """
        # Find which segment each input falls into
        # Clamp x to be within the range of knots
        x_clamped = torch.clamp(x, min=self.knots[0], max=self.knots[-1])

        # Find the segment each x belongs to
        # For each x, find the largest knot that is less than or equal to x
        segments = torch.searchsorted(self.knots, x_clamped) - 1
        segments = torch.clamp(segments, 0, self.num_breakpoints - 2)

        # Get the knots and values for each segment
        x0 = torch.gather(self.knots, 0, segments)
        x1 = torch.gather(self.knots, 0, segments + 1)
        y0 = torch.gather(self.values, 0, segments)
        y1 = torch.gather(self.values, 0, segments + 1)

        # Get the derivatives at the knots
        d0 = torch.gather(self.derivatives, 0, segments)
        d1 = torch.gather(self.derivatives, 0, segments + 1)

        # Normalize the x coordinate to [0, 1] within each segment
        h = x1 - x0
        t = (x_clamped - x0) / h

        # Compute cubic Hermite spline coefficients
        t2 = t * t
        t3 = t2 * t

        # Hermite basis functions
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        # Evaluate the cubic polynomial
        y = h00 * y0 + h10 * (h * d0) + h01 * y1 + h11 * (h * d1)

        return y