import torch
from torch import nn
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import Cosyne, PGPE
from evotorch.logging import StdOutLogger


class LinearPolicy(nn.Module):

    def __init__(
        self, 
        obs_length: int, # Number of observations from the environment
        act_length: int, # Number of actions of the environment
        bias: bool = True,  # Whether the policy should use biases
        **kwargs # Anything else that is passed
        ):
        super().__init__()  # Always call super init for nn Modules
        self.linear = nn.Linear(obs_length, act_length, bias = bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Forward pass of model simply applies linear layer to observations
        return self.linear(obs)


print('Train LinearPolicy with GymNE')

# Specialized Problem class for RL
problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    # Linear policy
    network=LinearPolicy,
    network_args = {'bias': False},  # Linear policy should not use biases
    num_actors= 4,  # Use 4 available CPUs. Note that you can modify this value, or use 'max' to exploit all available CPUs
    observation_normalization = False,  # Observation normalization was not used in Lunar Lander experiments
)

radius_init = 4.5  # (approximate) radius of initial hypersphere that we will sample from
max_speed = radius_init / 15.  # Rule-of-thumb from the paper
center_learning_rate = max_speed / 2.

searcher = PGPE(
    problem,
    popsize=200,  # For now we use a static population size
    radius_init= radius_init,  # The searcher can be initialised directely with an initial radius, rather than stdev
    center_learning_rate=center_learning_rate,
    stdev_learning_rate=0.1,  # stdev learning rate of 0.1 was used across all experiments
    optimizer="clipup",  # Using the ClipUp optimiser
    optimizer_config = {
        'max_speed': max_speed,  # with the defined max speed 
        'momentum': 0.9,  # and momentum fixed to 0.9
    }
)

StdOutLogger(searcher)
searcher.run(50)


center_solution = searcher.status["center"]  # Get mu
policy_net = problem.to_policy(center_solution)  # Instantiate a policy from mu
for _ in range(10):  # Visualize 10 episodes
    result = problem.visualize(policy_net)
    print('Visualised episode has cumulative reward:', result['cumulative_reward'])


print('Done!')
print('---------------------------------------')
print('---------------------------------------')
print('---------------------------------------')
print('Train LinearPolicy with CoSyNE')

problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    network=LinearPolicy,
    network_args = {'bias': False},
    num_actors= 4, 
    observation_normalization = False,
    num_episodes = 3,
    initial_bounds = (-0.3, 0.3),
)

searcher = Cosyne(
    problem,
    num_elites = 1,
    popsize=50,  
    tournament_size = 4,
    mutation_stdev = 0.3,
    mutation_probability = 0.5,
    permute_all = True, 
)

StdOutLogger(searcher)
searcher.run(50)

center_solution = searcher.status["pop_best"]  # Get the best solution in the population
policy_net = problem.to_policy(center_solution)  # Instantiate the policy from the best solution
for _ in range(10): # Visualize 10 episodes
    result = problem.visualize(policy_net)
    print('Visualised episode has cumulative reward:', result['cumulative_reward'])
