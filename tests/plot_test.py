from src.plotter import plot_loss_curve, plot_weight_path

# 1. Simulate dummy data for a typical optimization run
# The loss steadily decreases towards a minimum
dummy_loss = [1465.0, 900.5, 550.2, 300.1, 150.8, 90.3, 75.1, 71.0, 70.2]

# Because of normalization, the intercept (a0) will stay right around 0
dummy_a0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

# The slope (a1) climbs from our starting guess (0) up to the optimal value
dummy_a1 = [0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.83, 0.85, 0.854] 

print("Testing Plot Generation...")

# 2. Call the functions
plot_loss_curve(dummy_loss, "Gradient Descent")
plot_weight_path(dummy_a0, dummy_a1, "Gradient Descent")

print("\nSuccess! Check the 'plots' directory in your VS Code explorer.")