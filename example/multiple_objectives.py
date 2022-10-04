import torch
from evotorch import Problem
from evotorch.algorithms import SteadyStateGA
from evotorch.operators import (
    SimulatedBinaryCrossOver,
    GaussianMutation,
)
from evotorch.logging import StdOutLogger


# Kursawe function with two conflicting objectives
def kursawe(x: torch.Tensor) -> torch.Tensor:
    f1 = torch.sum(
        -10 * torch.exp(
            -0.2 * torch.sqrt(x[:, 0:2] ** 2.0 + x[:, 1:3] ** 2.0)
        ),
        dim=-1,
    )
    f2 = torch.sum(
        (torch.abs(x) ** 0.8) + (5 * torch.sin(x ** 3)),
        dim=-1,
    )
    fitnesses = torch.stack([f1, f2], dim=-1)
    return fitnesses

prob = Problem(
    # Two objectives, both minimization
    ["min", "min"],
    kursawe,
    initial_bounds=(-5.0, 5.0),
    solution_length=3,
    vectorized=True,
)

# Works like NSGA-II for multiple objectives
ga = SteadyStateGA(prob, popsize=200)
ga.use(
    SimulatedBinaryCrossOver(
        prob,
        tournament_size=4,
        cross_over_rate=1.0,
        eta=8,
    )
)
ga.use(GaussianMutation(prob, stdev=0.03))
logger = StdOutLogger(ga)

ga.run(100)
