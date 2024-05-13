import matplotlib.pyplot as plt
from framework.problem.problem import Model
import numpy as np
import torch

def plot_loss(loss_dict: dict, ax:plt.Axes=None):
    
    if ax == None:
        fig, ax = plt.subplots()

    ax.grid()
    
    color_options = [ "g", "r", "b"]
    line_options = [ "dashed", "dashdot", "dotted"]
    
    i = 0
    for key, value in loss_dict.items():
        if  key == "total_loss":
            ax.plot(range(len(value)), value, label=key, c="black", ls="solid", lw=4, alpha=0.5,zorder=-1)
        else:
            ax.plot(range(len(value)), value, label=key, c=color_options[i], ls=line_options[i], lw=2)
            i += 1
        
    ax.legend(fontsize=9, loc=1)
    ax.set_yscale('log')
    ax.set_xlabel("Optimization step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Loss",fontsize=16)

    return ax

def plot_function_values_over_t(model:Model, ax:plt.Axes=None):
   
    if ax == None: 
        fig, ax = plt.subplots()

    ax.grid()
    
    if model.analytical_fnc != None:
        t = torch.linspace(model.data.domain_dict["t"][0], model.data.domain_dict["t"][1], 1000)[:,None]
        ax.plot(t, model.analytical_fnc(t).T , label="Analytical", c="green", lw=4, alpha = 0.5)
        
    t = model.data.domain
    u = model.forward(t).detach().numpy()
    ax.plot(t.detach().numpy(), u , label="Q-ML", c="red", ls="dashdot", lw=2)
        
    ax.legend(fontsize=9, loc=1)
    ax.set_xlabel("t step", fontsize=13)
    ax.set_ylabel("U(t)", fontsize=13)
    ax.set_title("Values of Time",fontsize=16)