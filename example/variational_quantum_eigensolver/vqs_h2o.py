# This example code is based on the "PennyLane's Variational Quantum Eigensolver" problem.
# For more information about this problem, see https://pennylane.ai/qml/demos/tutorial_vqe.html

# In brief, a VQE consists of 3 components:
# 1. A parameterisable quantum circuit Q with p parameters, which prepares the ground state of the molecule.
# 2. A cost function C that computes the energy of a given ground state, which we want to minimise.
# 3. A classical optimisation algorithm, which searches for the optimal parameter vector which minimises the energy.

# Here we will be using variation quantum circuits for Q, which means that the circuits are parameterised by a vector of angles.


import pennylane as qml
from functools import partial
from evotorch import Problem, Solution
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
from typing import Optional
import torch


symbols = ["H", "O", "H"]  #H2O molecule
coordinates = torch.tensor([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])  # nuclear coordinates in atomic units

H, qubits = qml.qchem.molecular_hamiltonian(
    symbols, 
    coordinates,
    charge=0,
    mult=1,
    basis="sto-3g",
    active_electrons=4,
    active_orbitals=4,
) 

print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)


electrons = 10
orbitals = 7
core, active = qml.qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)

singles, doubles = qml.qchem.excitations(len(active), qubits)
hf = qml.qchem.hf_state(
    len(active), 
    qubits,
)  # giving the Hartree-Fock state 

# Map excitations to the wires the UCCSD circuit will act on
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

# Define the device
dev = qml.device('default.qubit', wires=qubits)

def circuit2(param, wires):
    # Setting up the circuit to optimise, which simply consists of preparing the basis state as the Hartree-Fock state 
    # And then applying a UCCSD ansatz
    qml.UCCSD(param, wires=wires, s_wires = s_wires, d_wires = d_wires, init_state = hf)

@qml.qnode(dev, diff_method=None)  # Disabling gradients -- we don't need them
def cost_fn(param):
    # Defining the cost function: simply apply the parameterised circuit and take the exponent of the Hamiltonian
    circuit2(param, wires=range(qubits))
    return qml.expval(H)


# Defining a new problem:
class VGEH2O(Problem):

    def __init__(self, num_actors: Optional[int]):

        super().__init__(
            objective_sense='min',  # Minimise the objective
            solution_length = 26,  # There are 26 parameters to optimise
            initial_bounds = (-1e-6, 1e-6),  # Start the search very close to zero
            num_actors = num_actors,
        )

    def _prepare(self):

        symbols = ["H", "O", "H"]  #H2O molecule
        coordinates = torch.tensor([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])
        self._H, self._qubits = qml.qchem.molecular_hamiltonian(
            symbols, 
            coordinates,
            charge=0,
            mult=1,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
        ) 

        electrons = 10
        orbitals = 7
        core, active = qml.qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)

        singles, doubles = qml.qchem.excitations(len(active), qubits)
        self._hf = qml.qchem.hf_state(
            len(active), 
            qubits,
        ) 

        self._s_wires, self._d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # Prepare function called by actors once instantaited
        dev = qml.device("default.qubit", wires=self._qubits)

        # Inline definition of cost function allows us to easily decorate it as a quantum node
        @qml.qnode(dev, diff_method = None, interface = 'torch')
        def actor_cost_fn(param):
            with torch.no_grad():
                wires = range(self._qubits)
                qml.UCCSD(param, wires=wires, s_wires = self._s_wires, d_wires = self._d_wires, init_state = self._hf)
                return qml.expval(self._H)

        self._cost_fn = actor_cost_fn

    def _evaluate(self, individual: Solution):
        x = individual.access_values()  # Get the decision values
        cost = self._cost_fn(x)  # Evaluate the parameter vector
        individual.set_evals(cost)  # Update the fitness

problem = VGEH2O(num_actors = 4)
population = problem.generate_batch(10)
problem.evaluate(population)


searcher = SNES(problem, stdev_init=0.1)  # stdev_init=0.1 used in [3]
pandas_logger = PandasLogger(searcher)
stdout_logger = StdOutLogger(searcher)

# Run for 200 generations
searcher.run(200)

progress = pandas_logger.to_dataframe()
progress.mean_eval.plot()
