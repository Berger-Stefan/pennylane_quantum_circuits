from sympy import re
import torch
import itertools
import pennylane as qml
    
class Data:
    def __init__(self,  domain_dict:dict) -> None:
        self.domain_dict = domain_dict
        self.build_domain()
    
    def build_domain(self):
        tmp = []
        for i in self.domain_dict.values():
            tmp.append(torch.linspace(i[0],i[1],i[2]))
        self.domain = torch.tensor([x for x in itertools.product(*tmp)], requires_grad=True)


class Model:
    def __init__(self, n_wires:int, trainable_params:dict, data:Data, embedding_ansatz, variational_ansatz, cost_function=None) -> None:
        self.embedding_ansatz = embedding_ansatz
        self.variational_ansatz = variational_ansatz
        
        self.data = data

        self.n_wires = n_wires

        self.device = qml.device("default.qubit", wires=n_wires)
        self.qnode = qml.QNode(self.circuit, self.device, diff_method="best")

        self.params_embedding = {} # TODO enter these values
        self.params_variation = {} #TODO enter these values
        self.set_embedding_parameter()

        self.trainable_params = trainable_params

        self.output_dim = len(trainable_params["weights"])

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
        output = []
        vcircuit = torch.vmap(self.qnode, in_dims=(0,None))
        for i in range(self.output_dim):
            output.append(self.trainable_params["scaling"][i]*vcircuit(input_values, self.trainable_params["weights"][i]) + self.trainable_params["bias"][i])
        if len(output) > 1:
            return torch.vstack(*output)
        else:
            return output[0]
    

class Solver:
    def __init__(self, data:Data, model:Model, pde_res_fnc, boundary_res_fnc, loss_scaling = None, optimizer="adam") -> None:
        self.data = data
        self.model = model
        self.loss_scaling = loss_scaling
        self.pde_res_fnc = pde_res_fnc
        self.boundary_res_fnc = boundary_res_fnc

        self.optimizer = optimizer
        if optimizer == "adam":
            self.opt = torch.optim.Adam(sum(self.model.trainable_params.values(), []), lr=0.1)
        elif optimizer == "lbfgs":
            pass

    
    def loss(self):
        # Compute PDE loss
        pde_loss_value = self.pde_res_fnc(self.model, self.data.domain)
        # Compute boundary loss
        boundary_losses  = []
        for boundary_fnc in self.boundary_res_fnc:
            boundary_losses.append(boundary_fnc(self.model,self.data.domain))
        
        if self.loss_scaling != None:
            total_loss = self.loss_scaling[0] * pde_loss_value + sum([ self.loss_scaling(idx)*val for idx,val in enumerate(boundary_losses)])
        else:
            total_loss = pde_loss_value + sum(boundary_losses)
        
        return total_loss

    def optimize(self, n_iter=1000):
        if self.optimizer == "adam":
            for i in range(1,n_iter+1):
                self.opt.zero_grad()
                loss = self.loss()
                loss.backward()
                self.opt.step()
                print(f"Step: {i}  Loss: {loss}")