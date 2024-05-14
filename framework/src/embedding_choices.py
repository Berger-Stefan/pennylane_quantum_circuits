import pennylane as qml
import torch

## 1D Embedding functions
def chebyshev_embedding(x, n_qubits, params):
    for i in range(n_qubits):
        qml.RY(2*torch.arccos(x),wires = i)

def chebyshev_rescaled_embedding(x, n_qubits, params):
    for i in range(n_qubits):
        rescaled_x = (x-params["x_0_min"])/(params["x_0_max"]-params["x_0_min"])*0.99
        qml.RY(2*torch.arccos(rescaled_x),wires = i)

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
def atan_embedding_alternating_2d(x, n_qubits, params):
    for i in range(n_qubits):
        if i%2 == 0:
            qml.RY(2*i*torch.atan(x[1]),wires = i)
        else:
            qml.RY(2*i*torch.atan(x[0]),wires = i)

def chebyshev_tower_embedding_rescaled_alternating_2d(x, n_qubits, params):
    for i in range(n_qubits):
        if i%2 == 0:
            rescaled_x = (x[0]-params["x_0_min"])/(params["x_0_max"]-params["x_0_min"])*0.99
            qml.RY(2*i*torch.acos(rescaled_x),wires = i)
        else:
            rescaled_x = (x[1]-params["x_1_min"])/(params["x_1_max"]-params["x_1_min"])*0.99
            qml.RY(2*i*torch.acos(rescaled_x),wires = i)
            
            
def chebyshev_tower_embedding_rescaled_parallel_2d(x, n_qubits, params):
    for i in range(n_qubits):
        if i >= round(n_qubits/2):
            rescaled_x = (x[0]-params["x_0_min"])/(params["x_0_max"]-params["x_0_min"])*0.99
            qml.RY(2*i*torch.acos(rescaled_x),wires = i)
        else:
            rescaled_x = (x[1]-params["x_1_min"])/(params["x_1_max"]-params["x_1_min"])*0.99
            qml.RY(2*i*torch.acos(x[1]),wires = i)

def chebyshev_tower_embedding_rescaled_sequential_2d(x, n_qubits, params):
    for i in range(n_qubits):
        rescaled_x = (x[0]-params["x_0_min"])/(params["x_0_max"]-params["x_0_min"])*0.99
        qml.RY(2*i*torch.acos(rescaled_x),wires = i)
        
    for i in range(n_qubits):
        rescaled_x = (x[1]-params["x_1_min"])/(params["x_1_max"]-params["x_1_min"])*0.99
        qml.RY(2*i*torch.acos(rescaled_x),wires = i)