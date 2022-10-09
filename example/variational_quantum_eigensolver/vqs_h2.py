# This example code is based on the "PennyLane's Variational Quantum Eigensolver" problem.
# For more information about this problem, see https://pennylane.ai/qml/demos/tutorial_vqe.html

# In brief, a VQE consists of 3 components:
# 1. A parameterisable quantum circuit Q with p parameters, which prepares the ground state of the molecule.
# 2. A cost function C that computes the energy of a given ground state, which we want to minimise.
# 3. A classical optimisation algorithm, which searches for the optimal parameter vector which minimises the energy.

# Here we will be using variation quantum circuits for Q, which means that the circuits are parameterised by a vector of angles.


import pennylane as qml
import torch
from evotorch import Problem, Solution
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
from typing import Optional



symbols = ["H", "H"]  #H2 molecule
coordinates = torch.tensor([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])  # nuclear coordinates in atomic units

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

dev = qml.device("default.qubit", wires=qubits)


electrons = 2  # H2 has 2 electrons
hf = qml.qchem.hf_state(electrons, qubits)  # giving the Hartree-Fock state 

def circuit(param, wires):
    # Setting up the circuit to optimise, which simply consists of preparing the basis state as the Hartree-Fock state 
    # And then applying a double excitation parameterised by the parameter
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

@qml.qnode(dev, diff_method=None)  # Disabling gradients -- we don't need them
def cost_fn(param):
    # Defining the cost function: simply apply the parameterised circuit and take the exponent of the Hamiltonian
    circuit(param, wires=range(qubits))
    return qml.expval(H)



class VGEMin(Problem):

    def __init__(self, num_actors: Optional[int] = None):

        super().__init__(
            objective_sense='min',  # Minimise the objective
            solution_length = 1,  # There is only a single parameter to optimise
            initial_bounds = (-1e-6, 1e-6),  # Start the search very close to zero
            num_actors = num_actors,  # Number of ray actors
        )

        symbols = ["H", "H"]
        coordinates = torch.tensor([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
        self._H, self._qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

        electrons = 2  # H2 has 2 electrons
        self._hf = qml.qchem.hf_state(electrons, qubits)

    def _prepare(self):
        # Prepare function called by actors once instantaited
        dev = qml.device("default.qubit", wires=self._qubits)

        # Inline definition of cost function allows us to easily decorate it as a quantum node
        @qml.qnode(dev, diff_method = None, interface = 'torch')
        def actor_cost_fn(param):
            with torch.no_grad():
                wires = range(self._qubits)
                qml.BasisState(self._hf, wires=wires)
                qml.DoubleExcitation(param[0], wires=[0, 1, 2, 3])
                return qml.expval(self._H)

        self._cost_fn = actor_cost_fn

    def _evaluate(self, individual: Solution):
        x = individual.access_values()  # Get the decision values -- in this case a vector of length 1
        cost = self._cost_fn(x)  # Evaluate the decision values
        individual.set_evals(cost)  # Update the fitness

problem = VGEMin(num_actors = 4)  # Instantiate the problem class
population = problem.generate_batch(5)  # Generate a population to test things out
problem.evaluate(population)  # If we've set everything up correctly we should get no error
print(f'Initial fitness values {population.access_evals()}')


searcher = SNES(problem, stdev_init=0.1)  # stdev_init=0.1 used in [3]
logger = PandasLogger(searcher)


searcher.run(100)

progress = logger.to_dataframe()
progress.mean_eval.plot()

print(f'Final mean is {searcher.status["center"]} Final stdev is {searcher.status["stdev"]}')
print(f'Cost of learned mean is {cost_fn(searcher.status["center"][0].numpy())} vs. approx global optima {cost_fn(0.208)}')
