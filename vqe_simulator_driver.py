from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from hamiltonian_creator import sample_hamiltonian


### test hamiltonian
hamiltonian = sample_hamiltonian()

# Define a sample variational ansatz
ansatz = EfficientSU2(hamiltonian.num_qubits)

estimator = Estimator()
optimizer = COBYLA()
vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)


# Display results
print("Ground state energy:", result.eigenvalue)
