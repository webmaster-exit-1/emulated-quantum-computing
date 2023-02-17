import numpy as np

def apply_gate(gate, target_qubit, state):
    """Applies a given quantum gate to a specific qubit or pair of qubits in a given state.

    Args:
        gate (array): The quantum gate to be applied.
        target_qubit (int or tuple): The index of the target qubit or pair of qubits to which the gate is applied.
        state (array): The state vector to which the gate is applied.

    Returns:
        array: The updated state vector after applying the gate.
    """
    state = np.array([1, 0, 0, 0])
    num_qubits = int(np.log2(len(state))) # Number of qubits in the state vector

    if isinstance(target_qubit, int):
        target_qubit = (target_qubit,)  # Convert target_qubit to a tuple with a single element
        # Single-qubit gate
        gate_tensor = np.eye(2)
        for i in range(1, num_qubits):
            if i == target_qubit:
                gate_tensor = np.kron(gate_tensor, gate)
            else:
                gate_tensor = np.kron(gate_tensor, np.eye(2))

    state_matrix = state.reshape((2 ** target_qubit[1], 2 ** (num_qubits - target_qubit[1])))
    assert 0 <= target_qubit[0] < num_qubits, f"Target qubit index must be between 0 and {num_qubits - 1}." # Check that the target qubit index is valid

    # Compute the tensor product of the gate with the identity matrix on all qubits except the target qubit
    gate_tensor = np.eye(2 ** (num_qubits - target_qubit[0] - 1))
    gate_tensor = np.kron(np.kron(np.eye(2 ** target_qubit[0]), gate), gate_tensor) # Tensor product of the gate with the identity matrix


    print("state shape before:", state.shape)
    state_matrix = state.reshape((2 ** target_qubit[1], 2 ** (num_qubits - target_qubit[1]))).T;
    print("state matrix shape:", state_matrix.shape)
    updated_state_matrix = gate_tensor.dot(state_matrix) # Compute the matrix product of the gate with the state matrix
    print("updated state matrix shape:", updated_state_matrix.shape)
    print("state shape after:", updated_state_matrix.flatten().shape)
    return updated_state_matrix.flatten() # Return the updated state vector

    # Apply the gate to the target qubit by computing the matrix product with the state vector
    state_matrix = state.reshape((2 ** target_qubit, 2 ** (num_qubits - target_qubit))) # Reshape the state vector into a matrix
    updated_state_matrix = gate_tensor.dot(state_matrix) # Compute the matrix product of the gate with the state matrix
    return updated_state_matrix.flatten() # Return the updated state vector
    isinstance(target_qubit, tuple) and len(target_qubit) == 2, "Target qubit must be an integer or a tuple of two integers."

    # Two-qubit gate
    assert gate.shape == (4, 4), "Gate must be a 4x4 matrix." # Check that the gate is a 4x4 matrix
    assert 0 <= target_qubit[0] < num_qubits and 0 <= target_qubit[1] < num_qubits and target_qubit[0] != target_qubit[1], "Target qubit indices must be between 0 and {num_qubits - 1} and different from each other." # Check that the target qubits are valid and different

    # Compute the tensor product of the gate with the identity matrix on all qubits except the target qubits
    gate_tensor = np.eye(2 ** (num_qubits - target_qubit - 1))
    gate_tensor = np.kron(np.kron(np.eye(2 ** (target_qubit[1] - target_qubit[0] - 1)), gate), gate_tensor) # Tensor product of the gate with the identity matrix
    gate_tensor = np.kron(np.kron(np.eye(2 ** target_qubit[0]), gate_tensor), np.eye(2 ** (num_qubits - target_qubit[1] - 1))) # Tensor product of the identity matrices with the gate_tensor

    # Apply the gate to the target qubit by computing the matrix product with the state vector
    gate_tensor = np.kron(gate, np.eye(2 ** (target_qubit[1] - target_qubit[0] - 1))) # Tensor product of the gate with the identity matrix
    gate_tensor = np.kron(np.eye(2 ** target_qubit[0]), gate_tensor) # Tensor product of the identity matrices and the gate
    gate_tensor = np.kron(gate_tensor, np.eye(2 ** (num_qubits - target_qubit[1] - 1))) # Tensor product of the gate with the identity matrix
    state_matrix = state.reshape((2 ** target_qubit[1], 2 ** (num_qubits - target_qubit[1]))) # Reshape the state vector
    state_tensor = np.kron(state_matrix, np.eye(2 ** target_qubit[0])) # Tensor product of the state matrix with the identity matrix
    updated_state_tensor = np.dot(gate_tensor, state_tensor.flatten()) # Compute the matrix product of the gate with the state tensor
    updated_state_matrix = updated_state_tensor.reshape((2 ** target_qubit[1], 2 ** target_qubit[0])) # Reshape the updated state tensor into a matrix
    # updated_state_matrix = gate_tensor.dot(state_matrix.flatten()) # Compute the matrix product of the gate with the state matrix
    #return updated_state_matrix.flatten() # Return the updated state vector
    return updated_state_matrix.reshape((2 ** target_qubit[1], 2 ** (num_qubits - target_qubit[1]))) # Return the updated state vector

