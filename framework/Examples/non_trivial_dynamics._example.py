# %%
import sys
sys.path.append("../../")

import framework
import torch
import math
from scipy.integrate import solve_ivp
torch.manual_seed(42)

# %% Data & Model
domain_dict = {"t":[0.001,0.9,1000]}
data = framework.Data(domain_dict)

n_wires = 8
n_layers = 8
weights = torch.ones((n_layers,n_wires), requires_grad=True)
bias = torch.ones(1, requires_grad=True)
scaling = torch.ones(1, requires_grad=True)
params = {"weights": [weights], "bias": [bias], "scaling": [scaling]}

embedding = framework.chebyshev_tower_embedding
variational = framework.basicEntanglerLayers

t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], 1000)
du_dt = lambda t, u: 4*u - 6*u**2 + math.sin(50*t) + u*math.cos(25*t) - 0.5
analytical_sol_fnc = solve_ivp(du_dt, [domain_dict["t"][0],domain_dict["t"][1]+0.000001], [0.75], t_eval=t, dense_output=True)

def analytical_fnc(input_values):
    return analytical_sol_fnc.sol(input_values)

model = framework.Model(n_wires, params, data, embedding, variational, analytical_fnc=analytical_fnc)

# %% Solver
def pde_res_fnc(model, t):
    u_pred = model.forward(t)
    grad_outputs = torch.ones_like(u_pred)
    du_dt_pred = torch.autograd.grad(u_pred, t, grad_outputs=grad_outputs, create_graph=True)[0]
    res = du_dt_pred - (4*u_pred - 6*u_pred**2 + torch.sin(50*t) + u_pred*torch.cos(25*t) - 0.5)
    return torch.mean(res**2)
 
def boundary_res_fnc(model):
    boundary_domain = torch.tensor([0.0], requires_grad=True)
    u_pred = model.forward(boundary_domain)
    return torch.mean((u_pred-0.75*torch.ones_like(u_pred))**2)


plot_update_functions = ["plot_loss", "plot_function_values_over_t"]
solver = framework.Solver(data, model,
                          pde_res_fnc, [boundary_res_fnc],
                          loss_scaling=[1,30],
                          plot_update_functions=plot_update_functions)

solver_settings_lbfgs = {"optimizer":"lbfgs", "learning_rate":1.0, "update_interval":1, "n_iter":10}
solver_settings_adam  = {"optimizer":"adam" , "learning_rate":0.05, "update_interval":10,"n_iter":100}
solver.optimize(solver_settings_adam)

# %%
