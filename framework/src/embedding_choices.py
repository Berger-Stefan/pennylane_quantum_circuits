import pennylane as qml
import torch

## 1D Embedding functions
def chebyshev_embedding(x, n_qubits, params):
    for i in range(n_qubits):
        qml.RY(2*torch.arccos(x),wires = i)

def chebyshev_rescaled_embedding(x, n_qubits, params):
    for i in range(n_qubits):
        qml.RY(2*torch.arccos((x-params["t_end"]/2)/params["t_end"]),wires = i)

def chebyshev_tower_embedding(x, n_qubits, params):
    for i in range(n_qubits):
        qml.RY(2*i*torch.arccos(x),wires = i)

def chebyshev_tower_rescaled_embedding(x, n_qubits, params):
    global t_end
    for i in range(n_qubits):
        qml.RY(2*i*torch.arccos((x-params["t_end"]/2)/params["t_end"]),wires = i)

def evolution_enchanted_embedding(x, n_qubits, params):
    pass

## 2D Embedding functions
def chebyshev_tower_embedding_alternating_2d(x, n_qubits, params):
    for i in range(n_qubits):
        if i%2 == 0:
            qml.RY(2*i*torch.atan(x[1]),wires = i)
        else:
            qml.RY(2*i*torch.atan(x[0]),wires = i)