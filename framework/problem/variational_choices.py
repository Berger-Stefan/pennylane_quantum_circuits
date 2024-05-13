import pennylane as qml
import torch

def basicEntanglerLayers(weights, n_wires):
    qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
        