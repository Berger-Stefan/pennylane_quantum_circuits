import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import framework
import torch
import numpy as np
import optuna


def advection_config_and_training_wrapper(trial):

    optimizer = "lbfgs"
    learning_rate = trial.suggest_float("learning rate", 0.5, 1.0, log=True)
    # optimizer = trial.suggest_categorical("optimizer", ["lbfgs", "adam"])

    t_sampling_points = trial.suggest_int("t_sampling_points", 10, 400)
    x_sampling_points = trial.suggest_int("x_sampling_points", 10, 400)

    boundary_initial_loss_scaling = trial.suggest_float("IV_loss_scaling", 0.1, 1000.0, log=True)
    boundary_sides_loss_scaling = trial.suggest_float("BC_loss_scaling", 0.1, 100.0, log=True)

    embedding_choice = trial.suggest_categorical(
        "embedding function",
        [
            "chebyshev_tower_embedding_rescaled_alternating_2d",
            "atan_embedding_alternating_2d",
            "chebyshev_tower_embedding_rescaled_parallel_2d",
            "chebyshev_tower_embedding_rescaled_sequential_2d",
        ],
    )
    variational_choice = trial.suggest_categorical("variational function", ["basicEntanglerLayers"])

    n_wires = trial.suggest_int("n_qubits", 4, 10)
    n_layers = trial.suggest_int("n_qubits", 4, 10)

    ## This part is the same as the code in the example
    advection_speed = 2

    domain_dict = {
        "t": [0.001, 0.999, t_sampling_points],
        "x": [-0.999, 0.999, x_sampling_points],
    }  # TODO check if include the boundary points
    data = framework.Data(domain_dict)

    weights = torch.ones((n_layers, n_wires), requires_grad=True)
    bias = torch.ones(1, requires_grad=True)
    scaling = torch.ones(1, requires_grad=True)
    params = {"weights": [weights], "bias": [bias], "scaling": [scaling]}

    embedding = getattr(framework, embedding_choice)
    variational = getattr(framework, variational_choice)

    def analytical_fnc(input_values):
        return -np.sin(np.pi * (input_values[:, 1] - advection_speed * input_values[:, 0]))

    model = framework.Model(n_wires, params, data, embedding, variational, analytical_fnc=analytical_fnc)

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
        input_domain_b1 = torch.stack((t, x), dim=1)
        u_pred_b1 = model.forward(input_domain_b1)

        t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], domain_dict["t"][2])
        x = -1 * torch.ones_like(t)
        input_domain_b2 = torch.stack((t, x), dim=1)
        u_pred_b2 = model.forward(input_domain_b2)
        return torch.mean((u_pred_b1 - u_pred_b2) ** 2)

    def bd3_fnc(model):
        t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], domain_dict["t"][2])
        x = torch.ones_like(t)
        input_domain_b1 = torch.stack((t, x), dim=1)
        u_pred_b1 = model.forward(input_domain_b1)

        t = torch.linspace(domain_dict["t"][0], domain_dict["t"][1], domain_dict["t"][2])
        x = -1 * torch.ones_like(t)
        input_domain_b2 = torch.stack((t, x), dim=1)
        u_pred_b2 = model.forward(input_domain_b2)
        return torch.mean((u_pred_b1 - u_pred_b2) ** 2)

    def bd4_fnc(model):
        x = torch.linspace(domain_dict["x"][0], domain_dict["x"][1], domain_dict["x"][2])
        t = torch.zeros_like(x)
        input_domain = torch.stack((t, x), dim=1)
        u_pred = model.forward(input_domain)
        return torch.mean((u_pred - (-torch.sin(torch.pi * x))) ** 2)

    def pde_res_fnc(model, input_values):
        u_pred = model.forward(input_values)

        grad_outputs_1 = torch.ones_like(u_pred)
        du = torch.autograd.grad(u_pred, input_values, grad_outputs=grad_outputs_1, create_graph=True)[0]
        du_dt_pred = du[:, 0]
        du_dx_pred = du[:, 1]

        du_du_dx = torch.autograd.grad(du_dx_pred, input_values, grad_outputs=grad_outputs_1, create_graph=True)[0]
        du_dx_dx_pred = du_du_dx[:, 1]

        res_pde = du_dt_pred - (-advection_speed * du_dx_pred)

        return torch.mean(res_pde**2)

    solver = framework.Solver(
        data,
        model,
        pde_res_fnc,
        [bd1_fnc, bd3_fnc, bd4_fnc],
        loss_scaling=[
            1,
            boundary_sides_loss_scaling,
            boundary_sides_loss_scaling,
            boundary_initial_loss_scaling,
        ],
    )

    if optimizer == "lbfgs":
        solver_settings_lbfgs = {
            "optimizer": "lbfgs",
            "learning_rate": learning_rate,
            "max_runtime": 5,
            "n_iter": 100,
        }
    elif optimizer == "adam":
        solver_settings_adam = {
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "max_runtime": 5,
            "n_iter": 1000,
        }
    else:
        raise ValueError("Optimizer not supported")

    solver.optimize_optuna(solver_settings_lbfgs, trial)
