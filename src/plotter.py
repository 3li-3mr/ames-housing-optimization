import matplotlib.pyplot as plt
import os

# Dynamically find the project root and the plots directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')

# Ensure the plots directory exists just in case it was deleted
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_loss_curve(loss_history, algorithm_name="Algorithm"):

    plt.figure(figsize=(8, 5))
    
    # Styling the line and markers
    plt.plot(loss_history, marker='o', linestyle='-', color='#1f77b4', markersize=5, linewidth=2)
    
    # Titles and labels
    plt.title(f"{algorithm_name}: Loss Optimization Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration Count", fontsize=12)
    plt.ylabel("Sum of Squared Errors (Loss)", fontsize=12)
    
    # Grid and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the file
    filename = f"{algorithm_name.replace(' ', '_').lower()}_loss.png"
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=300) # High resolution
    print(f"Saved loss plot to: {filepath}")
    plt.close() # Close the figure to free up memory


def plot_weight_path(a0_history, a1_history, algorithm_name="Algorithm"):

    plt.figure(figsize=(8, 5))
    
    # Plot the path the weights took
    plt.plot(a0_history, a1_history, marker='x', linestyle='-', color='#ff7f0e', markersize=6, linewidth=2)
    
    # Distinctly mark the start and end points
    plt.scatter(a0_history[0], a1_history[0], color='red', s=100, label='Start', zorder=5)
    plt.scatter(a0_history[-1], a1_history[-1], color='green', s=100, label='End (Optimum)', zorder=5)

    # Titles and labels
    plt.title(f"{algorithm_name}: Weight Path (a0 vs a1)", fontsize=14, fontweight='bold')
    plt.xlabel("Intercept (a0)", fontsize=12)
    plt.ylabel("Slope (a1)", fontsize=12)
    
    # Grid, legend, and layout
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the file
    filename = f"{algorithm_name.replace(' ', '_').lower()}_weights.png"
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Saved weight path plot to: {filepath}")
    plt.close()