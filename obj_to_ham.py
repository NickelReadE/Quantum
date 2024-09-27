from qiskit_optimization import QuadraticProgram


def obj_to_quad_prog(r, ESG, Sigma):
    # Define constants
    lambda_1 = 1.0  # Adjust this as needed
    lambda_2 = 1.0  # Adjust this as needed
    lambda_3 = 2.0  # Adjust this as needed

    # Define the Quadratic Program
    qp = QuadraticProgram()

    # Add binary variables (modify this based on the number of stocks)
    num_assets = len(r)
    for i in range(num_assets):
        qp.binary_var(name=f'x_{i}')

    # Define the linear terms
    linear_terms = lambda_1 * r - lambda_3 * ESG

    # Define the quadratic terms
    quadratic_terms = -lambda_2 * Sigma

    # Set the objective function
    qp.minimize(linear=linear_terms.tolist(), quadratic=quadratic_terms.tolist())

    return qp

