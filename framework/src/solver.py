from tkinter import W
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from matplotlib.gridspec import GridSpec
from IPython import display

from .data import Data
from .model import Model

class Solver:
    def __init__(self, data:Data, model:Model, pde_res_fnc, boundary_res_fnc:list, loss_scaling = None, plot_update_functions:list=[]) -> None:
        self.data = data
        self.model = model
        self.pde_res_fnc = pde_res_fnc
        self.boundary_res_fnc = boundary_res_fnc
        self.plot_update_functions = plot_update_functions
        self.loss_values = {"pde_loss":[], "boundary_loss":[], "total_loss":[]}
        if model.analytical_fnc != None:
            self.loss_values["analytical_loss"] = []

        if loss_scaling == None:
            self.loss_scaling = [1] + [ 1 for i in range(len(boundary_res_fnc))]
        else:
            self.loss_scaling = loss_scaling
    
    def closure(self):
        self.opt.zero_grad()
        loss = self.loss_fnc()
        loss.backward()
        return loss
    
    def loss_fnc(self):
        # Compute PDE loss
        pde_loss_value = self.pde_res_fnc(self.model, self.data.domain)
        # Compute boundary loss
        boundary_losses  = []
        for boundary_fnc in self.boundary_res_fnc:
            boundary_losses.append(boundary_fnc(self.model))
        
        total_loss = self.loss_scaling[0] * pde_loss_value + sum([ self.loss_scaling[idx+1]*val for idx,val in enumerate(boundary_losses)])

        # Save loss values
        self.loss_values["total_loss"].append(total_loss.detach().numpy())
        self.loss_values["pde_loss"].append(self.loss_scaling[0]*pde_loss_value.detach().numpy())
        self.loss_values["boundary_loss"].append(sum([ self.loss_scaling[idx+1]*val.detach().numpy() for idx,val in enumerate(boundary_losses)]))
        if self.model.analytical_fnc != None:
            res_analytical = np.mean((self.model.forward(self.data.domain).detach().numpy() - self.model.analytical_fnc(self.data.domain.detach().numpy().T))**2)
            self.loss_values["analytical_loss"].append(res_analytical)
        
        return total_loss

    def optimize(self, optimizer_settings_dict=None):

        # If no optimizer settings are passed, use default settings
        if optimizer_settings_dict == None:
            self.optimizer_settings_dict = {"optimizer":"adam" , "learning_rate":0.05, "update_interval":100,"n_iter":1000}
        
        self.optimizer_settings_dict = optimizer_settings_dict
        if optimizer_settings_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(sum(self.model.trainable_params.values(), []), lr=optimizer_settings_dict["learning_rate"])
        elif optimizer_settings_dict["optimizer"] == "lbfgs":
            self.opt = torch.optim.LBFGS(sum(self.model.trainable_params.values(), []), lr=optimizer_settings_dict["learning_rate"])
 
        for i in range(1,self.optimizer_settings_dict["n_iter"]+1):
            if self.optimizer_settings_dict["optimizer"] == "adam":
                self.opt.zero_grad()
                loss = self.loss_fnc()
                loss.backward()
                self.opt.step()
            elif self.optimizer_settings_dict["optimizer"] == "lbfgs":
                self.opt.step(self.closure)
                loss = self.loss_fnc()

            if i%self.optimizer_settings_dict["update_interval"] == 0: self.update(i, loss.detach().numpy())

            # if no improvement value is not lower then the lowest in last 100 steps
            if i > 100 and loss.item() - min(self.loss_values["total_loss"][-20:]) >= 1e-4:
                break
    
    def update(self, iteration:int, loss:float):
        print(f"Step: {iteration}  Loss: {loss}")
        
        n_plots = len(self.plot_update_functions)

        if n_plots == 0:
            return
        
        fig = plt.figure(layout="constrained")
        fig.set_figheight(5)
        fig.set_figwidth(20)

        gs = GridSpec(1, n_plots, figure=fig)

        # call the plot functions that are passed as strings
        for idx, plot_function in enumerate(self.plot_update_functions):
            ax = fig.add_subplot(gs[idx])
            getattr(self, plot_function)(ax)
            
        display.clear_output(wait=True)
        if not "PYTEST_CURRENT_TEST" in  os.environ:
            plt.show()
        print(f"Step: {iteration}  Loss: {loss}")
        
    def plot_loss(self, ax:plt.Axes=None):
        if ax == None:
            fig, ax = plt.subplots()

        ax.grid()
        
        color_options = [ "g", "r", "b"]
        line_options = [ "dashed", "dashdot", "dotted"]
        
        i = 0
        for key, value in self.loss_values.items():
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

    def plot_function_values_over_t(self, ax:plt.Axes=None):
        if ax == None: 
            fig, ax = plt.subplots()

        ax.grid()
        
        if self.model.analytical_fnc != None:
            t = torch.linspace(self.model.data.domain_dict["t"][0], self.model.data.domain_dict["t"][1], 1000)[:,None]
            ax.plot(t, self.model.analytical_fnc(t).T , label="Analytical", c="green", lw=4, alpha = 0.5)
            
        t = self.model.data.domain
        u = self.model.forward(t).detach().numpy()
        ax.plot(t.detach().numpy(), u , label="Q-ML", c="red", ls="dashdot", lw=2)
            
        ax.legend(fontsize=9, loc=1)
        ax.set_xlabel("t step", fontsize=13)
        ax.set_ylabel("U(t)", fontsize=13)
        ax.set_title("Values of Time",fontsize=16)

    def plot_2d_contour(self, ax:plt.Axes=None):
        if ax == None: 
            fig, ax = plt.subplots()

        ax.grid()
        
        x_1 = torch.linspace(self.data.domain_dict["t"][0], self.data.domain_dict["t"][1], 100)
        x_2 = torch.linspace(self.data.domain_dict["x"][0], self.data.domain_dict["x"][1], 100)
        domain_plot = torch.tensor([x for x in itertools.product(x_1,x_2)], requires_grad=True)
        u = self.model.forward(domain_plot)
        u = u.reshape(len(x_1), len(x_2)).detach().numpy()
        contourf = ax.contourf(x_1.detach().numpy(), x_2.detach().numpy(), u.T, levels=50, cmap="jet")
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('u', fontsize=13)  # Set label for the colorbar
        ax.set_xlabel("t", fontsize=13)
        ax.set_ylabel("x", fontsize=13)
        ax.set_title("2D Contour",fontsize=16)


        return ax