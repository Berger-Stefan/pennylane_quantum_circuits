import pytest

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import framework
import torch
from scipy.integrate import solve_ivp

torch.manual_seed(42)


def test_ode_system():
    domain_dict = {"t": [0.01, 0.9, 101]}
    data = framework.Data(domain_dict)

    n_wires = 4
    n_layers = 4
    weights_1 = torch.rand((n_layers, n_wires), requires_grad=True)
    weights_2 = torch.rand((n_layers, n_wires), requires_grad=True)
    bias_1 = torch.rand(1, requires_grad=True)
    bias_2 = torch.rand(1, requires_grad=True)
    scaling_1 = torch.rand(1, requires_grad=True)
    scaling_2 = torch.rand(1, requires_grad=True)
    params = {
        "weights": [weights_1, weights_2],
        "bias": [bias_1, bias_2],
        "scaling": [scaling_1, scaling_2],
    }

    embedding = framework.chebyshev_tower_embedding
    variational = framework.basicEntanglerLayers

    t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], 100)
    du_dt = lambda t, u: [5 * u[1] + 3 * u[0], -3 * u[1] - 5 * u[0]]
    analytical_sol_fnc = solve_ivp(
        du_dt,
        [domain_dict["t"][0], domain_dict["t"][1] + 0.000001],
        [0.5, 0.0],
        t_eval=t,
        dense_output=True,
    )

    def analytical_fnc(t):
        return analytical_sol_fnc.sol(t)

    model = framework.Model(n_wires, params, data, embedding, variational, analytical_fnc=analytical_fnc)

    # Boundary values
    #          bd1
    #         |---|
    #  ^ bd4  |   | bd2 (not used)
    #  |      |---|
    #  x       bd3
    #  t ->
    def bd_fnc(model):
        t = torch.tensor([0.0], requires_grad=True)
        u_pred_1, u_pred_2 = model.forward(t)

        u_0_1 = 0.5
        u_0_2 = 0
        return torch.mean((u_pred_1 - u_0_1) ** 2) + torch.mean((u_pred_2 - u_0_2) ** 2)

    def pde_res_fnc(model, t):
        u_pred_1, u_pred_2 = model.forward(t)

        grad_outputs_1 = torch.ones_like(u_pred_1)
        du_1_dt = torch.autograd.grad(u_pred_1, t, grad_outputs=grad_outputs_1, create_graph=True)[0]
        grad_outputs_2 = torch.ones_like(u_pred_2)
        du_2_dt = torch.autograd.grad(u_pred_2, t, grad_outputs=grad_outputs_2, create_graph=True)[0]

        res_1 = du_1_dt - 5 * u_pred_2 - 3 * u_pred_1
        res_2 = du_2_dt + 3 * u_pred_2 + 5 * u_pred_1
        return torch.mean(res_1**2) + torch.mean(res_2**2)

    solver = framework.Solver(
        data,
        model,
        pde_res_fnc,
        [bd_fnc],
        loss_scaling=[1, 20],
        plot_update_functions=["plot_loss", "plot_function_values_over_t"],
    )

    solver_settings_lbfgs = {
        "optimizer": "lbfgs",
        "learning_rate": 1.0,
        "update_interval": 1,
        "n_iter": 4,
    }
    solver.optimize(solver_settings_lbfgs)

    assert solver.loss_values["total_loss"][-1] < 1e-2
