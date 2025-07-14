import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


# # Physical parameters
beta = 2.2e9 
A = 0.95        
rho = 1000         
F = 50            
L = 24e3        
Nz = 200
h = L / Nz

NX = 10

g = 9.81

# Time span
t0, tmax = 0, 10

Nt = 50
t_eval = np.linspace(t0, tmax, Nt)

phi = lambda z : torch.zeros_like(z)

class trunk2(nn.Module):
    
    def __init__(self, layer_sizes, K, N, activation):
        super().__init__()
        
        
        self.branch = nn.Parameter(torch.ones((K,2)))
        
        self.K = K 
        self.N = N
        
        self.activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation] 
        
        layer_sizes = [1] + layer_sizes + [(2*self.N+1) * K]
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
        
    def basis(self, t):

        t = t.view(-1, 1)  # Ensure t is column vector of shape (T, 1)
        basis = [torch.ones_like(t)]  # Constant term

        for n in range(1, self.N + 1):
            basis.append(torch.cos(np.pi * n * t / tmax))
            basis.append(torch.sin(np.pi * n * t / tmax))

        return torch.cat(basis, dim=1)  # Shape: (T, 2N + 1)
        
    def forward(self, zt):
        
        t = zt[:,1][:Nt]
        z = zt[:,0].view(-1, Nt)[:, 0, None]
        
    
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
        z = self.linears[-1](z).view(-1, (2*self.N+1), self.K)
        
        t = self.basis(t)
        
        trunk = torch.einsum("zNK,tN->ztK", z, t)

        return torch.einsum("Kd,ztK->ztd", self.branch, trunk)
        
class Model2():
    def __init__(self, trunk, y):  
        
        #self.y = y
        self.z = torch.linspace(0, L, Nz, dtype=torch.float32)
        self.t = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True)
        self.zt = torch.cartesian_prod(self.z, self.t)   
        
        self.y = y
            
        self.net = trunk    
    
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.losshistory = []
        self.bestvloss = 1000000

    
    def dloss(self, output):
        
        
        output = output.reshape(Nz, Nt, -1)
                                
        y_pred = torch.stack([output[0,...], output[-1,...]], dim=0)
        
        r3 = self.mse(y_pred - self.y)
        
        return r3
    
    def ploss(self, output):
                                
        ddx = torch.stack(
            [
            torch.autograd.grad(
                output[..., i],
                self.zt,
                grad_outputs=torch.ones_like(output[..., 0]),
                create_graph=True
            )[0]
            for i in range(2)
            ],
            dim=-1
        ).reshape(Nz, Nt, 2, 2)
    
        
                
        ddz, ddt = ddx[:,:,0,:], ddx[:,:,1,:]    
        
        q = output[...,0]
        q_z, q_t = ddz[...,0], ddt[...,0]
        p_z, p_t = ddz[...,1], ddt[...,1]
                        

        r1 = self.mse(p_t + (beta / A) * q_z + (beta / A))
        
    
        r2 = self.mse(q_t + (A / rho) * p_z + (F / rho) * q + (1 / A)) # +  A * g * sin_theta(z) 
        
        return r1, r2
    
            
    def train(self, iterations, val_interval=100):
        
                
        self.steps = []
        self.net.train(True)
        print(f"{'step':<10} {'loss':<10} {'r1':<10} {'r2':<10} {'r3':<10} ")
        
        for iter in range(iterations):
                                    
            self.optimizer.zero_grad()
            output = self.net(self.zt)
                        
            r1, r2 = self.ploss(output)
            r3 = self.dloss(output)
            
            
            loss = r1 + r2
            loss.backward()
            self.optimizer.step()

            if iter % val_interval == val_interval - 1:
                self.net.eval()            
                loss = loss.item()

                if loss < self.bestvloss:
                    torch.save(self.net.state_dict(), "best_model.pth")    
                    self.bestvloss = loss                  
                    self.losshistory.append(loss)
                    self.steps.append(iter)
                    self.net.train(True)
                                
                fstring = f"{iter + 1:<10} {f'{r1+r2+r3:.2e}':<10} {f'{r1:.2e}':<10} {f'{r2:.2e}':<10} {f'{r3:.2e}':<10}"
                print(fstring)
                
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