import torch
import torch.nn as nn
from scipy.constants import g

import wntr
import pandas as pd
import numpy as np
import copy
import networkx as nx
import torch
from wntr.graphics import plot_network
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

import torch.nn.functional as F

from itertools import product



softplus = nn.Softplus()
relu = nn.ReLU()


def soft_sign(x, s=1e-3):
    # Smooth sign to avoid nondifferentiability at 0
    return torch.tanh(x / s)

def pow_clamped(z, p, eps=1e-12):
    # (z + eps)^p to avoid infinite slope at 0 for p<1
    return (z + eps).pow(p)


class Net(nn.Module):

    def __init__(self, layer_sizes, activation, base=None, positive=False):
        super().__init__()
        
        self.positive = positive
        self.base = base[:,None] if base is not None else None
        self.activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid(), 'softplus' : softplus}[activation] 
        #self.Dmax = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.Dmax = torch.tensor([0.5], dtype=torch.float32)
        
        layer_sizes = layer_sizes
        
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.linears.append(layer)
            
        
    def forward(self, x):
        
        
        #idx = x[...,-1].long()        
        
        for linear in self.linears[:-1]:

            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self.positive:
            x = softplus(x)
        #self.base[idx,:]
        return  x
    
    def predict(self, x):
        
        Q = self.forward(x)
        
        Q = Q[...,:Q.shape[-1]//2]
    

class Model(nn.Module):
    def __init__(self, model_params, net_params):  
        super().__init__()
        required_keys = ['A0', 'inv', 'M', 'B', 'a', 'S', 'demand_idx', 'L', 'd', 'Cd', 'C', 'rho', 'N']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
            
        
        #self.D = model_params['D']
        self.L = self.L[None,:]
        self.d = self.d[None,:]
        self.C = self.C[None,:]
        self.supply = (self.B @ self.S)[None,:]
        
        self.lambda_reg = 0.001
        
        self.register_buffer("I", torch.eye(self.A0.shape[1], dtype=self.A0.dtype, device=self.A0.device))
        
        self.n_pipes = self.M.shape[1]
        
        self.n_samples = model_params['n_samples']
        
        self.leak_id = torch.arange(self.n_pipes, dtype=torch.float32).reshape(-1,1)
        
        demand_x = torch.linspace(0, 1, self.N, dtype=torch.float32)
        self.D = torch.tensor(list(product(demand_x, repeat=len(self.demand_idx))), dtype=torch.float32)
        
        #self.D_eval = demand_mesh.repeat(self.n_pipes, 1)
        
        # self.D = torch.tensor(demand_mesh, dtype=torch.float32)
        self.ID = torch.arange(self.n_pipes).reshape(-1, 1)          
        D_exp  = self.D.unsqueeze(1).expand(-1, self.ID.shape[0], -1) 
        ID_exp = self.ID.unsqueeze(0).expand(self.D.shape[0], -1, -1)

        self.x_val = torch.cat([D_exp, ID_exp.float()], dim=-1)   
        
        self.Q_true = model_params['Q_true']
                    
        self.net = Net(**net_params)        
                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        
        self.bestloss = 1e12
                
        self.mse = lambda x : (x**2).mean()

    def hL(self, q):
        # This one is fine (p>1 → slope goes to 0 at 0)
        return torch.sign(q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * self.L * torch.abs(q)**(1.852)

    def hL_inv(self, x, eps=1e-12, s=1e-3):
        # Safe inverse: smooth sign and epsilon inside the fractional power
        K = self.C**(1.852) * self.d**(4.871)
        den = 10.667 * self.L
        z = (K * x.abs()) / (den + eps)          # magnitude input
        mag = pow_clamped(z, 1.0/1.852, eps=eps) # avoid ∞ slope at 0
        return soft_sign(x, s) * mag             # avoid sign’s hard kink

    def d_leak(self, a, H, eps=1e-12):
        # Option A: keep ReLU shape but stabilize sqrt near 0
        return self.Cd * a * torch.sqrt(2 * g * (F.relu(H) + eps))
    
    def mv(self, M, v):
        return (M @ v.T).T
    
    def prod(self, x, y):
           
        x_exp  = x.unsqueeze(1).expand(-1, y.shape[0], -1) 
        y_exp = y.unsqueeze(0).expand(x.shape[0], -1, -1)
        prod = torch.cat([x_exp, y_exp], dim=-1)
        return prod.reshape(-1, prod.shape[-1]) 
    
    def loss_(self,leak_id):
        
        demand = torch.zeros(self.n_samples, self.A0.shape[0])
        demand[:,self.demand_idx] = self.D.repeat(self.n_samples, 1)
        
        idx = leak_id.long()        
        batch_idx = torch.arange(self.n_samples).unsqueeze(1).expand_as(idx)
        
        areas = torch.zeros((self.n_samples, self.n_pipes))
        areas[batch_idx,:] = self.a
        
        H = self.net(leak_id)
        hL = self.supply - self.mv(self.A0.T, H)
    
        Q = self.hL_inv(hL)  
        

        loss = self.mse(self.mv(self.A0, Q) - demand - self.d_leak(self.mv(self.M, areas), H))

        return loss
    
    
    def loss(self, D):
        
        
        idx_exp = self.leak_id.repeat(D.shape[0] // self.n_pipes, 1)
        
        demand = torch.zeros(D.shape[0], self.A0.shape[0])
        demand[:,self.demand_idx] = D
        
        idx = idx_exp.long()        
        batch_idx = torch.arange(idx_exp.shape[0]).unsqueeze(1).expand_as(idx_exp)

        
        areas = torch.zeros((idx_exp.shape[0], self.n_pipes))
        areas.scatter_(1, idx, self.a[idx])

        input = torch.cat((D, idx_exp), dim=-1)
        
        
        Q = self.net(input)
        
        hL = self.hL(Q)  
        H = self.mv(self.inv, self.supply - hL) 
        #b = self.mv(self.A0, self.supply - hL)
        #y = torch.linalg.solve_triangular(self.L_chol, b.T, upper=False)
        #H = torch.linalg.solve_triangular(self.L_chol.T, y, upper=True).T
        loss1 = self.mse(self.mv(self.A0, Q) - demand - self.d_leak(self.mv(self.M, areas), H))
        loss2 = self.mse(self.supply - hL - self.mv(self.A0.T, H))
        
        return loss1, loss2

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.net.train(True)
        #print(f"{'step':<10} {'Q-loss':<10} {'H-loss':<10} {'sum-loss':<10}")
        
        p = 0.0
        iter = 0
        while p < 0.9:
            
            
            #idx = torch.arange(self.n_samples, dtype=torch.float32).reshape(-1,1)
            D = self.net.Dmax * torch.rand((self.n_samples*self.n_pipes, self.demand_idx.shape[0]))
            
                                                                        
            self.optimizer.zero_grad()
            loss1, loss2 = self.loss(D)
            loss = loss1 + loss2
            loss.backward()
            #all_grads = torch.cat([p.grad.view(-1) for p in self.net.parameters() if p.grad is not None])
            #print(all_grads)
            self.optimizer.step()
            
            #vloss = loss.item()
            
            if iter % print_interval == print_interval - 1:  
                p = self.validate() 
                #fstring = f"{iter + 1:<10} {f'{loss1.item():.2e}':<10} {f'{loss2.item():.2e}':<10} {f'{loss.item():.2e}':<10}"
                #print(fstring)
                             
                #if vloss < self.bestloss:
                #     torch.save(self.net.state_dict(), "best_model.pth")    
                #     self.bestloss = loss
                    
                    #announce = 'New Best!'
            iter += 1
            
    def validate(self):
        self.net.eval()
        with torch.no_grad():  

            Q = self.net(self.x_val)                         

            idx = (self.n_pipes + self.ID.unsqueeze(0).expand(Q.shape[0], -1, 1)).long() 
            Q_sel = Q.gather(dim=-1, index=idx)  
            Q_pred = torch.cat((Q[...,:Q.shape[-1]//2], Q_sel), dim=-1).detach().numpy() 
            
            rmse = np.sqrt(((Q_pred - self.Q_true) ** 2).mean(axis=-1))
            rng = self.Q_true.max(axis=-1) - self.Q_true.min(axis=-1)
            
            nrmse = rmse / rng
            
            p = float(np.mean(nrmse.reshape(-1) <= 0.1))
            
            print(f'p10={100*p:.2f}%', end="\r", flush=True)

            
            return p
            
        #print(((np.abs(Q_pred - self.Q_true)).mean(-1) / (self.Q_true.max(-1) - self.Q_true.min(-1))).mean())

                
    # def validate(self):
    #     self.net.eval()
    #     with torch.no_grad():
    #         loss1, loss2 = self.loss(self.D_eval)
    #         vloss = loss1 + loss2
    #         print(f'Validation loss = {vloss}')
    #     return vloss
    
    
    # def eval(self):
    #     print('Best loss:', self.bestloss)
    #     self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
    #     self.net.eval()
        
       