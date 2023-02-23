from quantemu import apply_gate
import numpy as np

# Apply the Hadamard gate to the first qubit
state = np.array([1, 0, 0, 0]) # Initialize the state vector
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2) # Hadamard gate
state = apply_gate(H, state, 0) # Apply the Hadamard gate to the first qubit

# Print the updated state vector
print(state)
