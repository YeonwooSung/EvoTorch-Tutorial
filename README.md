# EvoTorch Tutorial

Example codes & tutorials for EvoTorch, a library for evolutionary algorithms for PyTorch

[Quick Start](https://docs.evotorch.ai/v0.2.0/quickstart/)

## Installation

```bash
pip install evotorch
```

### Requirements

- Python 3.6+
- PyTorch 1.1+
- evotorch

**1. MNIST**

For the MNIST example, you will also need to install torchvision.

**2. Reinforcement Learning**

For the RL example, you will also need to install gym. Also, box2d, mujoco, and pygame should be installed.
To install the box2d, you need to install the swig first. On Ubuntu, you can install it by running:

```bash
sudo apt-get install swig
```

On macOS, you can install it by running:

```bash
brew install swig
```

After install the swig, you can install the box2d by running:

```bash
pip3 install box2d-py
```

If you still gets some unexpected errors due to the box2d, this might fix it:

```bash
pip3 install box2d box2d-kengz
```

## Usage

For general usage of v0.2.0, please refer [this link](https://docs.evotorch.ai/v0.2.0/user_guide/general_usage/).

### Basic Usage

```python
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger


def norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    problem = Problem(
        "min",
        norm,
        initial_bounds=(-10.0, 10.0),
        solution_length=100,
        vectorized=True,
        device=device, # For non-numeric problems, only CPU is supported. If it is numeric, other devices such as GPU could be used.
    )

    searcher = SNES(problem, popsize=1000, stdev_init=10.0)
    _ = StdOutLogger(searcher, interval=50)
    searcher.run(num_generations=1000)
```

### Advanced Usage

1. [GPU acceleration](./example/gpu_accelerate.py)

2. [Multiple Objectives](./example/multiple_objectives.py)

3. Reinforcement Learning with Evolutionary Algorithms
    - [Paralleized Reinforcement Learning](./example/rl/humanoid_v4_linear.py)
    - [LunarLanderContinuous-v2](./example/rl/lunarlander_continuous_v2.py)

4. [Train MNIST with SNES](./example/mnist_example.py)

5. [Variational Quantum Eigensolvers with SNES](./example/variational_quantum_eigensolver/)
    - [VQE solver for H2](./example/variational_quantum_eigensolver/vqs_h2.py)
    - [VQE solver for H2O](./example/variational_quantum_eigensolver/vqs_h2o.py)
