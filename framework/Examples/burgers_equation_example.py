# %%
import sys
sys.path.append("../../")

import framework
import torch

torch.manual_seed(42)

# %% Data & Model

domain_dict = {"t":[0,1.0,10],"x":[-1.,1.,10]}
data = framework.Data(domain_dict)

n_wires = 8
n_layers = 3
weights = torch.ones((n_layers,n_wires), requires_grad=True)
bias = torch.ones(1, requires_grad=True)
scaling = torch.ones(1, requires_grad=True)
params = {"weights": [weights], "bias": [bias], "scaling": [scaling]}

embedding = framework.chebyshev_tower_embedding_alternating_2d
variational = framework.basicEntanglerLayers

model = framework.Model(n_wires, params, data, embedding, variational)

# %% Solver

# Boundary values
#          bd1
#         |---|
#  ^ bd4  |   | bd2 (not used)
#  |      |---|
#  x       bd3
#  t ->
def bd1_fnc(model):
    t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], domain_dict["t"][2])
    x = torch.ones_like(t)
    input_domain = torch.stack((t,x), dim=1)
    u_pred = model.forward(input_domain)
    return torch.mean(u_pred**2) 

def bd3_fnc(model):
    t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], domain_dict["t"][2])
    x = -1*torch.ones_like(t)
    input_domain = torch.stack((t,x), dim=1)
    u_pred = model.forward(input_domain)
    return torch.mean(u_pred**2) 

def bd4_fnc(model):
    x = torch.linspace(domain_dict["x"][0], domain_dict["x"][1], domain_dict["x"][2])
    t = torch.zeros_like(x)
    input_domain = torch.stack((t,x), dim=1)
    u_pred = model.forward(input_domain)
    return torch.mean((u_pred - (-torch.sin(torch.pi*x)))**2) 

def pde_res_fnc(model, input_values):
    u_pred = model.forward(input_values)
    
    grad_outputs_1 = torch.ones_like(u_pred)
    du = torch.autograd.grad(u_pred, input_values, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dt_pred = du[:,0]
    du_dx_pred = du[:,1]
    
    du_du_dx = torch.autograd.grad(du_dx_pred, input_values, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dx_dx_pred = du_du_dx[:,1]
    
    res_pde = du_dt_pred # + u_pred * du_dx_pred - 0.01/torch.pi*du_dx_dx_pred

    return torch.mean(res_pde**2)

solver = framework.Solver(data, model,
                          pde_res_fnc, [bd1_fnc, bd3_fnc, bd4_fnc],
                          loss_scaling=[1,1,1,2],
                          plot_update_functions=["plot_loss", "plot_2d_contour"])

# %% Optimizer

solver_settings_lbfgs = {"optimizer":"lbfgs", "learning_rate":1.0, "update_interval":1, "n_iter":10}
solver_settings_adam  = {"optimizer":"adam" , "learning_rate":0.05, "update_interval":10,"n_iter":100}
solver.optimize(solver_settings_adam)
# %%
