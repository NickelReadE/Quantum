from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator as RuntimeEstimator

from hamiltonian_creator import sample_hamiltonian

# Note to team: (Phoebus)
#   1. Replace token with your own. Get one at qiskit quant computing platform.
#   2. This code isn't working:
#       - possible reason: the vqe is no longer supported by ibm, it is run by community.
#         so it doesn't take EstimatorV2 anymore (which provided by the runtime estimator).
#
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='')
backend = service.least_busy(operational=True, simulator=False)


with Session(service=service, backend=backend) as session:
    ### test hamiltonian
    hamiltonian = sample_hamiltonian()

    # Define a sample variational ansatz
    ansatz = EfficientSU2(hamiltonian.num_qubits)

    estimator = RuntimeEstimator(mode=session)
    optimizer = COBYLA()
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)


# Display results
print("Ground state energy:", result.eigenvalue)
