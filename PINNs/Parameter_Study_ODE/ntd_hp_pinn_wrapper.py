import deepxde as dde
from deepxde.backend import tf
import torch
import numpy as np
from scipy.integrate import solve_ivp
import datetime

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


    # Solver settings
    t_start = 0.0
    t_end   = 0.9
    u_0 = 0.75    
   
   # Define domain 
    geom = dde.geometry.TimeDomain(t_start, t_end)

    def boundary(_, on_initial):
        return on_initial

    ic = dde.icbc.IC(geom, lambda t: 0.75, boundary)
    
    def sol(t):
        derivatives_fnc = lambda t,u: 4*u[0] - 6*u[0]**2 + np.sin(50*t) + u[0]*np.cos(25*t) - 0.5
        analytical_solution = solve_ivp(derivatives_fnc, [t_start,t_end], [u_0], dense_output=True)
        return analytical_solution.sol(t).T
    def function_res(t, u):
        du_dt = dde.grad.jacobian(u, t)
        res = du_dt - (4*u[0] - 6*u[0]**2 + torch.sin(50*t) + u[0]*torch.cos(25*t) - 0.5)
        return res

    data = dde.data.PDE(geom, function_res, [ic], 100, 1, num_test=30)

    # Design the model structure
    layer_size = [1] + [50] * 4 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    model = dde.Model(data, net)
    model.compile("adam", lr=0.01)

    losshistory, train_state = model.train(iterations=20000)
    
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
