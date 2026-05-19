import sys
import os

# Allow imports from the src directory when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from data_prep import load_and_prep_data
from gradient_descent import run_and_plot as gd_run
from newton_method import run_and_plot as nm_run


def main():
    X, y = load_and_prep_data()

    print("=" * 60)
    print("  OPTIMIZATION COMPARISON: Gradient Descent vs Newton")
    print("=" * 60)

    # ── Gradient Descent runs ──────────────────────────────────
    print("\n[Experiment 1] Gradient Descent — alpha = 0.0001 (normal convergence)")
    gd_a0, gd_a1, gd_iter_normal = gd_run(
        X, y, learning_rate=0.0001, max_iter=5000,
        label="Gradient Descent (alpha=0.0001)"
    )

    print("\n[Experiment 2] Gradient Descent — alpha = 0.01 (too large / overshooting)")
    _, _, gd_iter_overshoot = gd_run(
        X, y, learning_rate=0.01, max_iter=1000,
        label="Gradient Descent (alpha=0.01 overshoot)"
    )

    # ── Newton's Method runs ───────────────────────────────────
    print("\n[Experiment 3] Newton's Method — alpha = 1.0 (full step)")
    nm_a0, nm_a1, nm_iter_full = nm_run(
        X, y, learning_rate=1.0, max_iter=100,
        label="Newton's Method (alpha=1.0)"
    )

    print("\n[Experiment 4] Newton's Method — alpha = 0.5 (damped step)")
    _, _, nm_iter_damped = nm_run(
        X, y, learning_rate=0.5, max_iter=100,
        label="Newton's Method (alpha=0.5 damped)"
    )

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    nm_full   = "Newton's Method (alpha=1.0)"
    nm_damped = "Newton's Method (alpha=0.5 damped)"
    print(f"  {'Algorithm':<42} {'Iterations':>10}  {'Status'}")
    print(f"  {'-'*42} {'-'*10}  {'-'*14}")
    print(f"  {'Gradient Descent (alpha=0.0001)':<42} {gd_iter_normal:>10}  Converged")
    print(f"  {'Gradient Descent (alpha=0.01)':<42} {gd_iter_overshoot:>10}  DIVERGED (NaN)")
    print(f"  {nm_full:<42} {nm_iter_full:>10}  Converged")
    print(f"  {nm_damped:<42} {nm_iter_damped:>10}  Converged")
    print("=" * 60)

    print("\n  Final weights comparison (GD alpha=0.0001 vs Newton alpha=1.0):")
    print(f"    Gradient Descent -> a0 = {gd_a0:.6f}, a1 = {gd_a1:.6f}")
    print(f"    Newton's Method  -> a0 = {nm_a0:.6f}, a1 = {nm_a1:.6f}")
    print()


if __name__ == "__main__":
    main()
