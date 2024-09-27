from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
import numpy as np

from obj_to_ham import obj_to_quad_prog


r_sample = np.array([0.5, 0.8, 0.3])  # Vector of returns
ESG_sample = np.array([0.4, 0.6, 0.7])  # Vector of ESG scores
Sigma_sample = np.array([[0.1, 0.02, 0.03],
              [0.02, 0.2, 0.04],
              [0.03, 0.08, 0.3]])  # Covariance matrix

qp = obj_to_quad_prog(r=r_sample, ESG=ESG_sample, Sigma=Sigma_sample)


print(qp.to_ising())

# Define the sampler and optimizer for QAOA
sampler = Sampler()
optimizer = COBYLA()  # You can choose a different optimizer if desired

# Set up QAOA with the required sampler and optimizer
qaoa = QAOA(sampler=sampler, optimizer=optimizer)

# Use MinimumEigenOptimizer to solve the problem
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

result = qaoa_optimizer.solve(qp)
print(f'Optimal value: {result.fval}')
print(f'Optimal solution: {result.x}')
