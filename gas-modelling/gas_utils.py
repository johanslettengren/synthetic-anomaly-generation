import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import itertools


from numpy import pi
from scipy.constants import g


class Net(nn.Module):

    def __init__(self, K, dim_out, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation
        self.networks = nn.ModuleList()
        
        layer_sizes = layer_sizes + [dim_out*K]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
        
    def forward(self, x):

        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)

        return x
        
        
class DeepONet(nn.Module):
    def __init__(self, dim_out, K, layer_sizes_trunk, layer_sizes_branch, activation='tanh'):
        super().__init__()

        activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid()}[activation] 
        
        self.K = K
        self.dim_out = dim_out
        self.trunk = Net(K=K, dim_out=1, layer_sizes=layer_sizes_trunk, activation=activation)
        self.branch = Net(K=K, dim_out=dim_out, layer_sizes=layer_sizes_branch, activation=activation)
        
        
    def forward(self, t, param):                

        a = self.trunk(t)
        B = self.branch(param).view(-1, self.K, self.dim_out)
        out = torch.einsum("pkd,tk->tpd", B, a)

        return out
    
    def derivative(self, t, param):                

        a = self.trunk(t)
        dadt = torch.cat([
            torch.autograd.grad(a[...,i], t, grad_outputs=torch.ones_like(a[...,0]), create_graph=True)[0] for i in range(self.K)
        ], dim=-1)
        
        
        B = self.branch(param).view(-1, self.K, self.dim_out)
        out = torch.einsum("pkd,tk->tpd", B, dadt).reshape(-1, self.dim_out)

        return out
    

class Model():
    def __init__(self, model_params, net_params, method='endpoint'):  
        
        required_keys = ['A0', 'd', 'd0', 'L', 'hL', 'hR', 'lambda0', 'T_max', 'Kl_max', 'n_samples']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
            
        self.S = pi * self.d**2 / 4
        
        self.A0_abs = torch.abs(self.A0)
        self.A0_R = ( self.A0 + torch.abs(self.A0) ) / 2
        
        self.Dp = - 1 / ( self.L * self.S )
        self.Dq =  - self.S / self.L
        self.Dg = g * (self.hR - self.hL)
        self.Df = self.lambda0 / (2 * self.d * self.S)
        
        self.pnet = DeepONet(self.A0.shape[0], **net_params)
        self.qnet = DeepONet(self.A0.shape[1], **net_params)
        
        params = itertools.chain(self.pnet.parameters(), self.qnet.parameters())

                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(params, lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.bestloss = 1e12
        
        self.loss = self.endpoint_loss if method=='endpoint' else self.midpoint_loss
        
        self.mse = lambda x : (x**2).mean()


    def mv(self, A, v):
        return (A @ v.T).T

    def midpoint_loss(self, t, p):
        pass
    
    def endpoint_loss(self, t, param):
        p = self.pnet(t, param)
        q = self.qnet(t, param)
        
        pdot = self.pnet.derivative(t, param)
        qdot = self.qnet.derivative(t, param)
        
        res1 = self.mv(self.A0_R, ( self.d0 * self.mv(self.A0_R.T, pdot) / self.Dp )) + self.mv(self.A0, q) # NEXT UP
        
        e1 = self.mse(res1)
        
        return e1, torch.tensor(0.0, requires_grad=True)

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.qnet.train(True)
        self.pnet.train(True)
        print(f"{'step':<10} {'loss':<10} {'r1':<10}  {'r2'}")
        
        for iter in range(iterations):
            t = self.T_max * torch.rand((self.n_samples, 1), requires_grad=True)
            p = self.Kl_max * torch.rand((self.n_samples, 1))
                                    
            self.optimizer.zero_grad()
            e1, e2 = self.loss(t, p)
            loss = e1 + e2
            loss.backward()
            self.optimizer.step()

            if iter % print_interval == print_interval - 1:           
                if loss < self.bestloss:
                    torch.save({
                        'pnet_state_dict': self.pnet.state_dict(),
                        'qnet_state_dict': self.qnet.state_dict(),
                    }, "best_model.pth")
                    self.bestloss = loss
                    self.pnet.train(True)
                    self.qnet.train(True)
                                
                fstring = f"{iter + 1:<10} {f'{loss:.2e}':<10} {f'{r1:.2e}':<10} {f'{r2:.2e}':<15}"
                print(fstring)
                
        print('Best loss:', self.bestloss)
        
        checkpoint = torch.load("best_model.pth")
        self.pnet.load_state_dict(checkpoint['pnet_state_dict'])
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.pnet.eval()
        self.qnet.eval()
