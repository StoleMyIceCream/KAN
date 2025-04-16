# Kolmogorov-Arnold Networks Implementation

This repository contains a PyTorch implementation of Kolmogorov-Arnold Networks (KANs) as described in the paper ["Kolmogorov-Arnold Networks: A Mathematical Framework for Representing Functions"](https://arxiv.org/abs/2404.19756).

## Installation

```bash
# Clone the repository
git clone https://github.com/stolemyicecream/kan-implementation.git
cd kan-implementation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `kan/`: Core implementation of KAN models
- `experiments/`: Python scripts for running experiments
- `notebooks/`: Jupyter notebooks for interactive exploration
- `utils/`: Utility functions for visualization and evaluation

## Tasks

This project addresses the following questions:

1. Build a minimal KAN
2. Apply KAN to fit the function f(x, y) = sin(xy) + cos(x² + y²)
3. Compare convergence and fit quality against a shallow MLP and other architectures
4. Analyze the loss surface of a shallow KAN with randomly initialized coefficients
5. Discuss implications for optimization dynamics

## Usage

### Using Python Scripts

```bash
# Run the minimal KAN implementation
python experiments/q1_minimal_kan.py

# Run function fitting experiment
python experiments/q2_function_fitting.py
```

### Using Jupyter Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook
```

Then navigate to the `notebooks/` directory and open the desired notebook.

## References

1. Liu, R., Jain, S., Gard, Y., & Drusvyatskiy, D. (2023). Kolmogorov-Arnold Networks. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
2. https://www.dailydoseofds.com/implementing-kans-from-scratch-using-pytorch/
3. https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/