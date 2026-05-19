from src.data_prep import load_and_prep_data, calculate_sse_loss, compute_gradients, compute_hessian

# 1. Test Data Loading
print("Loading data...")
X, y = load_and_prep_data('data/AmesHousing.csv')
print(f"Data loaded successfully! N={len(X)} records.")
print(f"Sample X (Normalized): {X[:3]}")
print(f"Sample y (Normalized): {y[:3]}")

# 2. Test Loss Function
initial_a0, initial_a1 = 0.0, 0.0
loss = calculate_sse_loss(X, y, initial_a0, initial_a1)
print(f"\nInitial Loss (a0=0, a1=0): {loss:.2f}")

# 3. Test Derivatives (For your teammates)
grads = compute_gradients(X, y, initial_a0, initial_a1)
hessian = compute_hessian(X)

print(f"\nCalculated Gradients: {grads}")
print(f"Calculated Hessian:\n{hessian}")