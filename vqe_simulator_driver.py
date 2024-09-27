from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
import numpy as np

from hamiltonians import sample_hamiltonian, objective_hamiltonian


# Callback function to print intermediate results for each iteration
def callback(nfev, parameters, energy, sdandev):
    """
    This callback function is passed into VQE for printing iterative outcome.
    :param nfev:
    :param parameters:
    :param energy:
    :param sdandev:
    :return:
    """
    # Construct the ansatz circuit with the current parameters
    param_dict = dict(zip(ansatz.parameters, parameters))
    ansatz_bound = ansatz.assign_parameters(param_dict)

    # Simulate the circuit to get the statevector
    from qiskit.quantum_info import Statevector
    state = Statevector(ansatz_bound)

    # Get the eigenvector (statevector data)
    eigenvector = state.data

    # Reverse the statevector to match the desired qubit ordering
    eigenvector_reversed = _reverse_statevector(eigenvector, hamiltonian.num_qubits)

    # Print the iteration number, eigenvalue, and eigenvector
    print(f"Iteration {nfev}:")
    print(f"  Eigenvalue: {energy}")
    print(f"  Eigenvector: {eigenvector_reversed}\n")


def _reverse_statevector(statevector, num_qubits):
    """
    This function handles reversing the ordering of qiskit statevector
    :param statevector:
    :param num_qubits:
    :return:
    """
    permuted_indices = [int(bin(i)[2:].zfill(num_qubits)[::-1], 2) for i in range(len(statevector))]
    return statevector[permuted_indices]


if __name__ == '__main__':
    # tickers and ESG scores of them can be put here
    tickers = ['NEE', 'ENPH', 'FSLR', 'BEP', 'SPWR', 'PLUG', 'ORA', 'CWEN', 'AES', 'SEDG', 'BE', 'AY', 'HASI', 'CSIQ', 'DQ', 'RUN', 'XEL']
    ESG = np.array([85, 78, 80, 83, 75, 77, 79,82, 76, 81, 74, 79, 80, 85, 76, 82, 86])

    # Define hamiltonian
    hamiltonian = objective_hamiltonian(tickers=tickers, ESG=ESG, weights=[1, 2, 3])

    # Define a sample variational ansatz
    ansatz = EfficientSU2(num_qubits=len(tickers))

    estimator = Estimator()
    optimizer = COBYLA()
    vqe = VQE(estimator, ansatz, optimizer, callback=callback)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

    # Display final results
    print("Ground state energy:", result.eigenvalue)
