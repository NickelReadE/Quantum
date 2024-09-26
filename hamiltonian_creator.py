from qiskit.quantum_info import SparsePauliOp


def sample_hamiltonian():
    """
    A test hamiltonian
    :rtype: SparsePauliOp
    """
    # Encode the hamiltonian
    pauli_terms = [
        ('ZZ', 1.0),
        ('ZI', 0.5),
        ('IZ', 0.25),]
    hamiltonian = SparsePauliOp.from_list(pauli_terms)
    return hamiltonian
