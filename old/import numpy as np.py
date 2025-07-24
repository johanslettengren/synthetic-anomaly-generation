import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


# # Physical parameters
beta = 1 #2.2e9 
A = 1 #0.95        
rho = 1 #1000         
F = 1 #50            
L = 1 #24e3        
Nz = 200
h = L / Nz

NX = 10

g = 9.81

# Time span
t0, tmax = 0, 10

Nt = 50
t_eval = np.linspace(t0, tmax, Nt)

phi = lambda z : torch.zeros_like(z)

class trunk(nn.Module):
    
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
        layer_sizes = [2] + layer_sizes + [K]
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def fwd(self, zt):
        for linear in self.linears[:-1]:
            zt = self.activation(linear(zt))
        return self.linears[-1](zt)
        
        
        
    def forward(self, zt):
        azt = self.fwd(zt)
        return azt
        
class Model():
    def __init__(self, net):  
        
        
        self.z = torch.linspace(0, L, Nz, dtype=torch.float32)
        self.t = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True)
        self.zt = torch.cartesian_prod(self.z, self.t)   
            
        self.net = net      
    
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.losshistory = []
        self.bestvloss = 1000000

    
    def ploss(self, eps, epsX):
        
        d = self.decoder(eps).T[:,None,None,:]        
                
        q_z, p_z, q_t, p_t, q = self.net.derivative(self.zt, epsX)      
                        
        ends_zt = torch.cartesian_prod(torch.tensor([0, L], dtype=torch.float32), self.zt[:,1][:Nt])
        ends = self.net(ends_zt, epsX) 
        X = self.X[None,None,...,None]
        r1 = self.mse(p_t + (beta / A) * q_z + (beta / A) * d * X)
        
    
        r2 = self.mse(q_t + (A / rho) * p_z + (F / rho) * q + (1 / A) * d * X**2) # +  A * g * sin_theta(z) 
        
        return r1, r2, ends
    
            
    def train(self, iterations, val_interval=100):
        
                
        self.steps = []
        self.net.train(True)
        print(f"{'step':<10} {'loss':<10} {'r1':<10}  {'r2'}")
        
        for iter in range(iterations):
            
            eps = torch.randn(self.Neps, 4)            
            n = self.X.shape[0]
            X_expanded = self.X.repeat_interleave(self.Neps).unsqueeze(1)
            eps_expanded = eps.repeat(n, 1)
            epsX = torch.cat([X_expanded, eps_expanded], dim=1)
                                    
            self.optimizer.zero_grad()
            r1, r2, _ = self.dloss(eps, epsX)
            loss = r1 + r2
            loss.backward()
            self.optimizer.step()

            if iter % val_interval == val_interval - 1:
                self.net.eval()            
                loss = loss.item()
                   
                # dql, dp0 = ends[1,...,0], ends[1,...,1]
                # new = self.y + ends
                # dql_min, dql_max = dql.min(), dql.max()
                # dp0_min, dp0_max = dp0.min(), dp0.max()
                # ql, p0 = new[1,...,0], new[1,...,1]
                # ql_min, ql_max = ql.min(), ql.max()
                # p0_min, p0_max = p0.min(), p0.max()
            
                
                # Check if we have a new best validation loss
                # announce_new_best = ''
                if loss < self.bestvloss:
                    
                    # announce_new_best = 'New best model!'
                    torch.save(self.net.state_dict(), "best_model.pth")    
                    self.bestvloss = loss                  
                    self.losshistory.append(loss)
                    self.steps.append(iter)
                    self.net.train(True)
                                
                fstring = f"{iter + 1:<10} {f'{r1+r2:.2e}':<10} {f'{r1:.2e}':<10} {f'{r2:.2e}':<15}"
                print(fstring)

                # (    
                #     f"{iter + 1:<10} {f'{r1+r2:.2e}':<10} {f'{r1:.2e}':<10} {f'{r2:.2e}':<15}"
                #     f"{f'[{dql_min:.1e}, {dql_max:.1e}]':<27}"
                #     f"{f'[{dp0_min:.1e}, {dp0_max:.1e}]':<27}"
                #     f"{f'[{ql_min:.1e}, {ql_max:.1e}]':<27}"
                #     f"{f'[{p0_min:.1e}, {p0_max:.1e}]':<25}"
                #     f"{announce_new_best}"
                # )
                
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        
        
    def plot_losshistory(self, dpi=100):
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.plot(self.steps, self.losshistory, '.-', label='Loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()   
        
        
def generate_leaks(grid, num_leaks, random_state):
    
    grid = grid[:,None]

    #kernel = Matern(length_scale=0.1, nu=2.0)
    kernel = RBF(length_scale=0.1)
    
    dx = grid[1,0] - grid[0,0]

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    
    f = gp.sample_y(grid, n_samples=num_leaks, random_state=random_state) 
    
    f = f - np.percentile(f, 85, axis=0, keepdims=True)        

    f = (f + np.abs(f)) / 2
    f = f / (np.sum(f, axis=0) * dx)
        
    return torch.tensor(f, dtype=torch.float32).T