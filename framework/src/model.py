import torch
import pennylane as qml

from .data import Data

class Model:
    def __init__(self, n_wires:int, trainable_params:dict, data:Data, embedding_ansatz, variational_ansatz, cost_function=None, analytical_fnc=None) -> None:
        self.embedding_ansatz = embedding_ansatz
        self.variational_ansatz = variational_ansatz
        self.data = data
        self.n_wires = n_wires
        self.device = qml.device("default.qubit", wires=n_wires)
        self.qnode = qml.QNode(self.circuit, self.device, diff_method="best")
        self.trainable_params = trainable_params
        self.output_dim = len(trainable_params["weights"])
        self.analytical_fnc = analytical_fnc
        self.params_embedding = {} # TODO enter these values
        self.params_variation = {} #TODO enter these values
        self.set_embedding_parameter()

    def set_embedding_parameter(self):
        self.params_embedding["t_end"] = self.data.domain_dict["t"][1]
    

    def circuit(self, input_values, weights):
        # Embedding
        self.embedding_ansatz(input_values, self.n_wires, self.params_embedding)
        #Variational ansatz
        self.variational_ansatz(weights, self.n_wires)
        # Cost function
        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(self.n_wires)]))
    

    def forward(self, input_values):
        vcircuit = torch.vmap(self.qnode, in_dims=(0,None))

        if self.output_dim > 1:
            output = []
            for i in range(self.output_dim):
                output.append(self.trainable_params["scaling"][i] * vcircuit(input_values, self.trainable_params["weights"][i]) + self.trainable_params["bias"][i])
            return output
        else:
            return self.trainable_params["scaling"][0] * vcircuit(input_values, self.trainable_params["weights"][0]) + self.trainable_params["bias"][0]