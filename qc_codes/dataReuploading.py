import pennylane as qml
from pennylane import numpy as pnp

### Circuit params
n_qubits = 8
n_layers = 3

### Circuit build 
dev = qml.device("default.qubit", wires=n_qubits)

def reupload_block(x, params):
    """
        One reuploading layer: weighted data encoding + entangling cascade.

        params has shape (n_qubits, 4): (w_y, b_y, w_z, b_z) per qubit.
        iff optional variational layer is used, params has shape (n_qubits, 7): (w_y, b_y, w_z, b_z, w_rx, w_ry, w_rz) per qubit.
    """
    for i in range(n_qubits):
        qml.RY(params[i, 0] * x[i] + params[i, 1], wires=i)
        qml.RZ(params[i, 2] * x[i] + params[i, 3], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
    ### optional variational layer after entanglement
    for i in range(n_qubits):
        qml.RX(params[i, 4], wires=i)
        qml.RY(params[i, 5], wires=i)
        qml.RZ(params[i, 6], wires=i)
        
@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    # weights shape: (n_layers, n_qubits, 4)
    for layer in range(n_layers):
        reupload_block(x, weights[layer])
    # return qml.expval(qml.PauliZ(0))      # not sure what best to measure
    # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return (
        [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        +
        [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1) % n_qubits))
        for i in range(n_qubits)]
    )