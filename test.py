import framework
import torch
import pennylane as qml
import matplotlib.pyplot as plt

# data
domain_dict = {"t":[0,1,10]}
data = framework.Data(domain_dict)

n_wires = 4
weights = torch.ones((3,n_wires), requires_grad=True)
bias = torch.ones(1, requires_grad=True)
scaling = torch.ones(1, requires_grad=True)
params = {"weights": [weights], "bias": [bias], "scaling": [scaling]}

embedding = framework.chebyshev_embedding
variational = framework.basicEntanglerLayers

model = framework.Model(n_wires, params, data, embedding, variational)
print(model.forward(data.domain[0,:]))

def pde_res_fnc(model, domain_data):
    u_pred = model.forward(domain_data)
    return torch.mean((u_pred - torch.ones_like(u_pred))**2)
 
def boundary_res_fnc(model, domain):
    boundary_domain = torch.tensor([0.0], requires_grad=True)
    u_pred = model.forward(boundary_domain)
    return torch.mean(u_pred**2)

solver = framework.Solver(data, model, pde_res_fnc, [boundary_res_fnc])

solver.optimize()