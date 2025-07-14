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

def solve(q0, pl, d, X):
    
    # initial conditions
    def initial_conditions():
        p_init = np.ones(Nz) * pl(0)
        q_init = np.ones(Nz) * q0(0)
        return np.concatenate([q_init, p_init])

    # RHS of ODE system
    def pq_rhs(t, y):
        
        q = y[:Nz]
        p = y[Nz:]
        
        q = np.append(q0(t), q)
        p = np.append(p, pl(t))
        
        p_rhs = - (beta / A) * (q[1:] - q[:-1]) / h - (beta / A) * d * X        
        q_rhs = - (A / rho) * (p[1:] - p[:-1]) / h - (F / rho) * q[1:] - (1 / A) * d * X**2

        return np.concatenate([q_rhs, p_rhs])

    # Solve the system
    y0 = initial_conditions()

    return solve_ivp(pq_rhs, [t0, tmax], y0, t_eval=t_eval, method='RK45')


class branch(nn.Module):
    """Postive Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
        self.K = K
        
        self.networks = nn.ModuleList()
        
        layer_sizes = [5] + layer_sizes + [2*K]
                
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
    
    
    def fourier_embed(self, eps):
        n_frequencies = 4
        B = 2**torch.arange(n_frequencies).float() * np.pi  # (n_freq,)
        eps_proj = eps.unsqueeze(-1) * B  # shape: [batch, dim_eps, n_freq]
        eps_proj = eps_proj.view(eps.shape[0], -1)  # flatten
        return torch.cat([torch.sin(eps_proj), torch.cos(eps_proj)], dim=1)  # shape: [batch, 2 * dim_eps * n_freq]


        
    def forward(self, x):
        # eps = x[:,:-1]
        # X = x[:,-1:]
        
        # eps = self.fourier_embed(eps) 
        
        # x = torch.cat([eps, X], dim=1)
            
        ox = x
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x).view(-1, self.K, 2)
        
        ox = torch.cat((ox[..., :-1], torch.zeros_like(ox[..., -1:])), dim=-1)
        for linear in self.linears[:-1]:
            ox = self.activation(linear(ox))
        ox = self.linears[-1](ox).view(-1, self.K, 2)
        
        return x - ox + 1
    
class onetrunk(nn.Module):
    
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
        
        
        
    def forward(self, x):
        
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        
        # ox = torch.zeros_like(x)
        # for linear in self.linears[:-1]:
        #     ox = self.activation(linear(ox))
        # ox = self.linears[-1](ox)
            
            
            
        return x
         

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
        
        t = zt[:,1][:Nt]
        z = zt[:,0].view(-1, Nt)[:, 0]
        
        n = int(zt.shape[0]/Nt)
        _t = torch.cartesian_prod(torch.zeros(n, dtype=torch.float32), t)
        z_ = torch.cartesian_prod(z, torch.zeros(Nt, dtype=torch.float32))
        zeros = torch.zeros_like(zt)
        
        azt = self.fwd(zt)
        a0t = self.fwd(_t)
        az0 = self.fwd(z_)
        a00 = self.fwd(zeros)
                    
        return azt - az0 - a0t + a00
    
    
class trunk2(nn.Module):
    
    def __init__(self, layer_sizes, K, activation, N=50):
        super().__init__()
        
        
        self.branch = nn.Parameter(torch.ones((K,2)))
        
        self.K = K 
        self.N = N
        
        self.activation = activation
        
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
        
    def fwd(self, zt):
        
        t = zt[:,1][:Nt]
        z = zt[:,0].view(-1, Nt)[:, 0, None]
        
    
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
        z = self.linears[-1](z).view(-1, (2*self.N+1), self.K)
        
        t = self.basis(t)
        output = torch.einsum("zNK,tN->ztK", z, t)
        
        
        return output.reshape(-1, self.K)
    
    def forward(self, zt):
        
        t = zt[:,1][:Nt]
        z = zt[:,0].view(-1, Nt)[:, 0]
        
        n = int(zt.shape[0]/Nt)
        _t = torch.cartesian_prod(torch.zeros(n, dtype=torch.float32), t)
        z_ = torch.cartesian_prod(z, torch.zeros(Nt, dtype=torch.float32))
        zeros = torch.zeros_like(zt)
        
        azt = self.fwd(zt)
        a0t = self.fwd(_t)
        az0 = self.fwd(z_)
        a00 = self.fwd(zeros)
                    
        return azt - az0 - a0t + a00

class DeepONet(nn.Module):
    def __init__(self, layer_sizes_trunk, layer_sizes_branch, K, activation):
        super().__init__()

        activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation] 
        
        self.K = K
        self.trunk = trunk(layer_sizes_trunk, K, activation)
        self.branch = branch(layer_sizes_branch, K, nn.ReLU())
        
        
    def forward(self, zt, epsX):                
        n = int(zt.shape[0]/Nt)
        trunk_output = self.trunk(zt)
        #basis = self.basis(zt)
        branch_output = self.branch(epsX)
        output = torch.einsum("ekd,xk->xed", branch_output, trunk_output)

        return output.view(n, Nt, -1, 2)
    

    
    def derivative(self, zt, epsX):
        
        trunk_output  = self.trunk(zt)
        branch_output = self.branch(epsX)
        
        dadx = torch.stack(
            [
            torch.autograd.grad(
                trunk_output[..., i],
                zt,
                grad_outputs=torch.ones_like(trunk_output[..., 0]),
                create_graph=True
            )[0]
            for i in range(self.K)
            ],
            dim=-1
        )
        
        n = int(zt.shape[0]/Nt)
        
        dadz, dadt = dadx[:,0,:], dadx[:,1,:]  
        
        output = torch.einsum("ekd,xk->xed", branch_output, trunk_output).view(n, Nt, -1, 2)
        ddz = torch.einsum("ekd,xk->xed", branch_output, dadz).view(n, Nt, -1, 2)
        ddt = torch.einsum("ekd,xk->xed", branch_output, dadt).view(n, Nt, -1, 2)
        
        q_z = ddz[...,0].reshape(n, Nt, NX, -1)
        p_z = -ddz[...,1].flip(dims=(0,)).reshape(n, Nt, NX, -1)
        
        q_t = ddt[...,0].reshape(n, Nt, NX, -1)
        p_t = ddt[...,1].flip(dims=(0,)).reshape(n, Nt, NX, -1)
        
        q = output[...,0].reshape(n, Nt, NX, -1)

        return q_z, p_z, q_t, p_t, q
        
class Model():
    def __init__(self, net, decoder):  
        
        
        self.z = torch.linspace(0, L, Nz, dtype=torch.float32)
        self.t = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True)
        self.zt = torch.cartesian_prod(self.z, self.t)
        self.X = torch.linspace(0, 1, NX, dtype=torch.float32)
        
        #self.X = torch.tensor([1], dtype=torch.float32)
        
        self.Neps = 10
        
            
        self.net = net      
        self.decoder = decoder  
    
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.losshistory = []
        self.bestvloss = 1000000

    
    def dloss(self, eps, epsX):
        
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