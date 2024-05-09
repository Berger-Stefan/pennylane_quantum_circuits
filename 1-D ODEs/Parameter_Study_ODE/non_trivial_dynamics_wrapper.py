from tkinter import Variable
import pennylane as qml
import torch
import torch.autograd as autograd
import math
import optuna
from datetime import datetime
from scipy.integrate import solve_ivp
import embedding_choices

def config_and_training_wrapper(trial):
    """
    Wrapper function for configuring and training a model using Optuna.

    Args:
        trial (optuna.Trial): The Optuna trial object.

    Returns:
        float: The minimum loss of the analytical solution.

    Raises:
        optuna.TrialPruned: If the trial is pruned.

    Description:
        This function configures and trains a model using Optuna. It takes a trial object from Optuna and uses it to suggest
        hyperparameters for the model. The function then sets up the necessary variables and initializes the model. It defines
        the loss functions for the differential and boundary conditions.
    """
    start_time = datetime.now()
    device = "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tunable Parameter
    n_qubits = 5
    #n_qubits = trial.suggest_int("n_qubits", 4,4)
    #n_layers = 5
    n_layers = trial.suggest_int("n_layers", 1, 5)
    learning_rate = 0.075
    #learning_rate = trial.suggest_float("learning rate", 0.05, 0.1, log=True)
    boundary_scaling = 5
    # boundary_scaling = trial.suggest_float("boundary scaling", 1.0, 20, log=False)
    n_steps = 100
    # n_steps = trial.suggest_int("collocation points", 1e1, 1e5, log=True)

    embedding_choice = trial.suggest_categorical("embedding function", ["chebyshev_embedding", "chebyshev_rescaled_embedding", "chebyshev_tower_embedding", "chebyshev_tower_rescaled_embedding"])
    embedding_fnc = getattr(embedding_choices, embedding_choice)

    # Solver settings
    t_start = 0.0001
    t_end   = 0.9
    t = torch.linspace(t_start,t_end,n_steps,requires_grad=True, device=device)  

    param_embedding = {"t_end": t_end}
    
    u_0 = torch.tensor(0.75)
    
    def derivatives_fnc(t, u):
        if isinstance(t, torch.Tensor):
            du_dt = 4*u - 6*u**2 + torch.sin(50*t) + u*torch.cos(25*t) - 0.5
        else:
            du_dt = 4*u - 6*u**2 + math.sin(50*t) + u*math.cos(25*t) - 0.5
        return du_dt

    analytical_solution = torch.tensor(solve_ivp(derivatives_fnc, [t_start,t_end+0.000001], [u_0], t_eval=t.detach()).y, device=device)

    weights = [torch.rand((n_layers, n_qubits), requires_grad=True, device=device)]        
    biases = [torch.rand(1, requires_grad=True, device=device)]
    scaling = [torch.rand(1, requires_grad=True, device=device)]

    parameters = weights + biases + scaling

    # Create optimizer
    opt = torch.optim.Adam(parameters, lr=learning_rate)
    loss_history = []
    loss_analytical_history = []


    @qml.qnode(qml.device("default.qubit.torch", wires=n_qubits), diff_method="backprop")
    def circuit(x, weights):
        # Embedding
        embedding_fnc(x,n_qubits, param_embedding)
        # Variational ansatz
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
        # Cost function
        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_qubits)]))

    def my_model(t, weights, bias, scaling):
        vcircuit = torch.vmap(circuit, in_dims=(0,None))
        return scaling[0]*vcircuit(t, weights[0]) + bias[0]

    def loss_diff_fnc(t:torch.Tensor, weights:list[torch.Tensor], biases:list[torch.Tensor], scaling:list[torch.Tensor]) ->torch.Tensor:
        u_pred = my_model(t, weights, biases, scaling) 

        grad_outputs_1 = torch.ones_like(u_pred)
        du_dt_pred = autograd.grad(u_pred, t, grad_outputs=grad_outputs_1, create_graph=True)[0]

        # ODE loss
        du_dt = derivatives_fnc(t, u_pred)
        res = du_dt_pred - du_dt
        loss_pde = torch.mean(res**2)
        return loss_pde

    def loss_boundary_fnc(t:torch.Tensor, weights:list[torch.Tensor], biases:list[torch.Tensor], scaling:list[torch.Tensor]) ->torch.Tensor:
        u_0_pred = my_model(torch.zeros_like(t), weights, biases, scaling)
        loss_boundary = torch.mean((u_0_pred - u_0)**2) 
        return boundary_scaling * loss_boundary

    def loss_fnc(t:torch.Tensor, weights:list[torch.Tensor], biases:list[torch.Tensor], scaling:list[torch.Tensor]) ->torch.Tensor:

        loss_diff     = loss_diff_fnc(t, weights, biases, scaling)
        loss_boundary = loss_boundary_fnc(t, weights, biases, scaling)

        return loss_boundary + loss_diff

    def loss_analytical_fnc(weights, biases, scaling):
        t = torch.linspace(t_start,t_end,n_steps, requires_grad=True)
        u_pred = my_model(t, weights, biases, scaling)
        return torch.mean((u_pred - analytical_solution)**3)
    
    for i in range(1,201):
        opt.zero_grad()
        loss = loss_fnc(t,weights, biases, scaling)
        loss.backward()
        opt.step()
        loss_history.append(loss.detach())
        loss_analytical_history.append(loss_analytical_fnc(weights, biases, scaling).detach())
        
        # Stop Training after 5 minutes
        if (datetime.now() - start_time).total_seconds() > (10*60): return min([float(tensor) for tensor in loss_analytical_history])
        
        trial.report(loss, i)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return min([float(tensor) for tensor in loss_analytical_history])
