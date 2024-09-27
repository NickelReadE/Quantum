from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
import numpy as np

from get_stock_data import get_r_and_Sigma


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


def sample_objective_to_hamiltonian():
    """
    for testing purposes
    :return:
    """
    # Define constants
    lambda_1 = 1.0  # Adjust this as needed
    lambda_2 = 1.0  # Adjust this as needed
    lambda_3 = 2.0  # Adjust this as needed

    # Define samples
    r = np.array([1, 2, 3])
    ESG = np.array([56, 46, 35])
    Sigma = np.array([[1, 2, 3], [1, 2, 3], [5, 3, 4]])

    # Define the Quadratic Program
    qp = QuadraticProgram()

    # Add variable for number of stocks
    num_assets = len(r)
    for i in range(num_assets):
        qp.binary_var(name=f'x_{i}')

    # Define the linear terms
    linear_terms = lambda_1 * r - lambda_3 * ESG

    # Define the quadratic terms
    quadratic_terms = -lambda_2 * Sigma

    # Set the objective function
    qp.minimize(linear=linear_terms.tolist(), quadratic=quadratic_terms.tolist())
    ising_ham = qp.to_ising()

    return ising_ham


def objective_hamiltonian(tickers, ESG, weights):
    """
    This function takes the tickers and ESG scores of companies, and fomulates the Ising hamiltonian
    It does so by stating the objective function, and turned into an hamiltonian.
    :param tickers: a np array of tickers for companies
    :param ESG: a np array of esg score, in the same order as tickers provided
    :param weight: a list of 3 numbers handling weight of these terms: return, sigma, ESG
    :return: Ising hamiltonian
    :rtype: SparsePauliOp
    """
    # Define constants
    lambda_1 = weights[0]  # Adjust this as needed
    lambda_2 = weights[1]  # Adjust this as needed
    lambda_3 = weights[2]  # Adjust this as needed

    # Define samples
    r, Sigma = get_r_and_Sigma(tickers)

    # Define the Quadratic Program
    qp = QuadraticProgram()

    # Add variable for number of stocks
    num_assets = len(r)
    for i in range(num_assets):
        qp.binary_var(name=f'x_{i}')

    # Define the linear terms
    linear_terms = -lambda_1 * r - lambda_3 * ESG

    # Define the quadratic terms
    quadratic_terms = lambda_2 * Sigma

    # Set the objective function
    qp.minimize(linear=linear_terms.tolist(), quadratic=quadratic_terms.tolist())
    ising_ham = qp.to_ising()
    return ising_ham[0]
