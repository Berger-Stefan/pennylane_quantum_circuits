import pennylane as qml
import torch
import torch.autograd as autograd
import math
import optuna

def config_and_training_wrapper(trial):

    device = "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tunable Parameter
    n_qubits = trial.suggest_int("n_qubits", 6, 6)
    n_layers = trial.suggest_int("n_layers", 1, 10)
    learning_rate = trial.suggest_float("learning rate", 0.01, 0.2, log=True)
    learning_rate = 0.2
    boundary_scaling = trial.suggest_float("boundary scaling", 0.1, 1e6, log=True)
    
    # Solver settings
    t_start = 0.0001
    t_end   = 0.9
    n_steps = trial.suggest_int("collocation points", 1e1, 1e5, log=True)
    t = torch.linspace(t_start,t_end,n_steps,requires_grad=True, device=device)  
    
    u_0 = torch.tensor(0.75)    

    weights = [torch.rand((n_layers, n_qubits), requires_grad=True, device=device)]        
    biases = [torch.rand(1, requires_grad=True, device=device)]
    scaling = [torch.rand(1, requires_grad=True, device=device)]

    parameters = weights + biases + scaling

    # Create optimizer
    opt = torch.optim.Adam(parameters, lr=learning_rate)
    loss_history = []
    
    def derivatives_fnc(t, u):
        if isinstance(t, torch.Tensor):
            du_dt = 4*u - 6*u**2 + torch.sin(50*t) + u*torch.cos(25*t) - 0.5
        else:
            du_dt = 4*u - 6*u**2 + math.sin(50*t) + u*math.cos(25*t) - 0.5
        return du_dt



    @qml.qnode(qml.device("default.qubit.torch", wires=n_qubits), diff_method="best")
    def circuit(x, weights):
        # Embedding
        for i in range(n_qubits):
            #qml.RY(2*i*torch.arccos((x-t_end/2)/(t_end)),wires = i)
            qml.RY(2*i*torch.arccos(x),wires = i)
        
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

    for i in range(1,201):
        opt.zero_grad()
        loss = loss_fnc(t,weights, biases, scaling)
        loss.backward()
        opt.step()
        loss_history.append(loss.detach())
        
        trial.report(loss, i)

        if trial.should_prune():
            raise optuna.TrialPruned()



    return min([float(tensor) for tensor in loss_history])