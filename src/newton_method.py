import numpy as np
from data_prep import load_and_prep_data, calculate_sse_loss, compute_gradients, compute_hessian
from plotter import plot_loss_curve, plot_weight_path


def newton_method(X, y, learning_rate=1.0, max_iter=100, tol=1e-6):

    # Initialise weights at zero
    a0 = 0.0
    a1 = 0.0

    loss_history = []
    a0_history   = []
    a1_history   = []

    # The Hessian only depends on X, so it is constant — compute it once
    H = compute_hessian(X)

    for iteration in range(max_iter):
        # Record current state
        loss = calculate_sse_loss(X, y, a0, a1)
        loss_history.append(loss)
        a0_history.append(a0)
        a1_history.append(a1)

        # Compute the gradient vector J
        J = compute_gradients(X, y, a0, a1)

        # Check convergence: stop when gradient magnitude is tiny
        if np.linalg.norm(J) < tol:
            print(f"[Newton's Method] Converged after {iteration} iterations.")
            return a0, a1, loss_history, a0_history, a1_history, iteration

        # Newton step: solve H * step = J  =>  step = H^-1 * J
        # Using np.linalg.solve is numerically safer than explicitly inverting H
        step = np.linalg.solve(H, J)

        # Damped parameter update
        a0 = a0 - learning_rate * step[0]
        a1 = a1 - learning_rate * step[1]

    print(f"[Newton's Method] Reached max iterations ({max_iter}).")
    return a0, a1, loss_history, a0_history, a1_history, max_iter


def run_and_plot(X, y, learning_rate=1.0, max_iter=100, label="Newton's Method"):
    """
    Runs Newton's method and saves both plots (loss curve + weight path).
    """
    a0, a1, loss_hist, a0_hist, a1_hist, n_iter = newton_method(
        X, y, learning_rate=learning_rate, max_iter=max_iter
    )

    print(f"  Final weights  ->  a0 = {a0:.6f},  a1 = {a1:.6f}")
    print(f"  Final SSE loss ->  {loss_hist[-1]:.6f}")
    print(f"  Iterations     ->  {n_iter}")

    plot_loss_curve(loss_hist, algorithm_name=label)
    plot_weight_path(a0_hist, a1_hist, algorithm_name=label)

    return a0, a1, n_iter


# ─── Experiments ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_and_prep_data()

    # Experiment 1: Full Newton step (alpha = 1.0)
    print("\n=== Experiment 1: alpha = 1.0 (full Newton step) ===")
    run_and_plot(X, y, learning_rate=1.0, max_iter=100,
                 label="Newton's Method (alpha=1.0)")

    # Experiment 2: Damped step (alpha = 0.5) — safer when far from optimum
    print("\n=== Experiment 2: alpha = 0.5 (damped / safer) ===")
    run_and_plot(X, y, learning_rate=0.5, max_iter=100,
                 label="Newton's Method (alpha=0.5 damped)")
