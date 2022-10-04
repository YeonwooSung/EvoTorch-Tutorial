# EvoTorch Tutorial

Example codes & tutorials for EvoTorch, a library for evolutionary algorithms for PyTorch

## Installation

```bash
pip install evotorch
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

3. [Paralleized Reinforcement Learning](./example/parallelized_rl.py)
