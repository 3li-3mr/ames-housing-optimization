import numpy as np
from data_prep import load_and_prep_data, calculate_sse_loss, compute_gradients


def gradient_descent(X, y, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Multi-Variable Gradient Descent for Linear Regression.

    Update rule:
        a0(k+1) = a0(k) - alpha * dL/da0
        a1(k+1) = a1(k) - alpha * dL/da1

    Parameters
    ----------
    X             : normalized feature array
    y             : normalized target array
    learning_rate : step size alpha
    max_iter      : maximum number of iterations
    tol           : convergence tolerance on the gradient norm

    Returns
    -------
    a0, a1        : learned parameters
    loss_history  : SSE loss at every iteration
    a0_history    : a0 value at every iteration
    a1_history    : a1 value at every iteration
    n_iter        : actual number of iterations until convergence
    """

    # Initialise weights at zero
    a0 = 0.0
    a1 = 0.0

    loss_history = []
    a0_history   = []
    a1_history   = []

    for iteration in range(max_iter):
        # Record current state
        loss = calculate_sse_loss(X, y, a0, a1)
        loss_history.append(loss)
        a0_history.append(a0)
        a1_history.append(a1)

        # Compute gradients
        gradients = compute_gradients(X, y, a0, a1)
        grad_a0, grad_a1 = gradients[0], gradients[1]

        # Check convergence: stop when gradient magnitude is tiny
        if np.sqrt(grad_a0**2 + grad_a1**2) < tol:
            print(f"[Gradient Descent] Converged after {iteration} iterations.")
            return a0, a1, loss_history, a0_history, a1_history, iteration

        # Parameter update
        a0 = a0 - learning_rate * grad_a0
        a1 = a1 - learning_rate * grad_a1

    print(f"[Gradient Descent] Reached max iterations ({max_iter}).")
    return a0, a1, loss_history, a0_history, a1_history, max_iter