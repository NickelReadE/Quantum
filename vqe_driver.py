from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
import numpy as np

from hamiltonians import objective_hamiltonian

# This is defined for coding convenience
cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],}


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy


if __name__ == '__main__':
    # IBM service connection
    service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main',
    token='623357116bf40ae972db60e06cb35610de97fbbb900c10f7cfc6ba5c88a2b851d04b47360f0a9c277de660dbed7a5b213e6fc0613a905cbf1574dac8c69f709c')
    backend = service.least_busy(operational=True, simulator=False)

    # tickers and ESG scores of them can be put here
    tickers = ['NEE', 'ENPH', 'FSLR', 'BEP', 'SPWR', 'PLUG', 'ORA', 'CWEN', 'AES', 'SEDG', 'BE', 'AY', 'HASI', 'CSIQ', 'DQ', 'RUN', 'XEL']
    ESG = np.array([85, 78, 80, 83, 75, 77, 79,82, 76, 81, 74, 79, 80, 85, 76, 82, 86])

    # Define hamiltonian
    hamiltonian = objective_hamiltonian(tickers=tickers, ESG=ESG, weights=[1, 2, 3])

    # Define a sample variational ansatz
    ansatz = EfficientSU2(hamiltonian.num_qubits)
    num_params = ansatz.num_parameters
    x0 = 2 * np.pi * np.random.random(num_params)

    # Optimize circuit
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)

    ansatz_isa = pm.run(ansatz)

    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    with Session(service=service, backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 100

        res = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="cobyla",
        )

        print(f'The eigenvector is {res.x}')
