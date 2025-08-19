import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import itertools


from numpy import pi
from scipy.constants import g

softplus = nn.Softplus()


class Net(nn.Module):

    def __init__(self, dim_out, layer_sizes, activation, positive=False):
        super().__init__()
        
        self.positive = positive
        self.activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid()}[activation] 
        self.networks = nn.ModuleList()
        
        
        layer_sizes = layer_sizes + [dim_out]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            #nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
        
    def forward(self, x):

        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)

        if self.positive:
            x = softplus(x)
        return x
        
        
# class DeepONet(nn.Module):
#     def __init__(self, dim_out, K, layer_sizes_trunk, layer_sizes_branch, activation='tanh', positive=False):
#         super().__init__()

#         activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid()}[activation] 
        
#         self.K = K
#         self.dim_out = dim_out
#         self.trunk = Net(K=K, dim_out=1, layer_sizes=layer_sizes_trunk, activation=activation, positive=positive)
#         self.branch = Net(K=K, dim_out=dim_out, layer_sizes=layer_sizes_branch, activation=activation, positive=positive)
        
        
#     def forward(self, t, param):                

#         a = self.trunk(t)
#         B = self.branch(param).view(-1, self.K, self.dim_out)
#         out = torch.einsum("pkd,tk->tpd", B, a)

#         return out
    
#     def derivative(self, t, param):                

#         a = self.trunk(t)
#         dadt = torch.cat([
#             torch.autograd.grad(a[...,i], t, grad_outputs=torch.ones_like(a[...,0]), create_graph=True)[0] for i in range(self.K)
#         ], dim=-1)
        
        
#         B = self.branch(param).view(-1, self.K, self.dim_out)
#         out = torch.einsum("pkd,tk->tpd", B, dadt)

#         return out
    

class Model():
    def __init__(self, model_params, net_params, method='endpoint'):  
        
        required_keys = ['A0', 'sp', 'Bs', 'dq', 'Bd', 'd', 'L', 'hL', 'hR', 'a', 'f', 'T_max', 'Kl_max', 'n_samples']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
            
                    
        self.A0_abs = torch.abs(self.A0)
        self.A0_R = ( self.A0 + torch.abs(self.A0) ) / 2
        
        self.Dh = self.hR - self.hL
        
        self.pnet = Net(self.A0.shape[0], **net_params, positive=True)
        self.qnet = Net(self.A0.shape[1], **net_params)
        
        params = itertools.chain(self.pnet.parameters(), self.qnet.parameters())

                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(params, lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.bestloss = 1e12
        
        self.loss = self.endpoint_loss if method=='endpoint' else self.midpoint_loss
        
        self.mse = lambda x : (x**2).mean()


    def mv(self, A, v):
        v_ = v.reshape(-1, v.shape[-1])
        return (A @ v_.T).T.reshape(v.shape[0], v.shape[1], -1)

    def midpoint_loss(self, t, p):
        pass
    
    def endpoint_loss(self, param):
        
        p = self.pnet(param)
        q = self.qnet(param)
        
        # pdot = self.pnet.derivative(param)
        # qdot = self.qnet.derivative(param)
                
        #t1 = self.mv(self.A0_R, self.L * self.mv(self.A0_R.T, pdot)) 
        t2 = - self.a**2 * (self.mv(self.A0, q) +  param * torch.sqrt(p)) / g
        t3 = self.a**2 * self.Bd @ self.dq / g
        t4 = - g * self.mv(self.A0_R, self.Dh * q)
        
        s1 = g * (-self.mv(self.A0.T, p) + self.Bs.T @ self.sp) 
        s2 =  10.67 * q**(1.852) / (self.f**(1.852) * self.d**(4.87)) 
        
        e1 = self.mse(t2 + t3 + t4)
        e2 = self.mse(s1 + s2)
        
        return e1, e2

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.qnet.train(True)
        self.pnet.train(True)
        print(f"{'step':<10} {'loss':<10} {'e1':<10}  {'e2'}")
        
        for iter in range(iterations):
            t = self.T_max * torch.rand((self.n_samples, 1), requires_grad=True)
            p = self.Kl_max * torch.rand((self.n_samples, 1))
                                    
            self.optimizer.zero_grad()
            e1, e2 = self.loss(p)
            
            if iter % 2 == 0:
                loss = e1
            else: 
                loss = e2
                
            vloss = e1 + e2
            loss.backward()
            self.optimizer.step()
            
            announce = ''
            if iter % print_interval == print_interval - 1:           
                if vloss < self.bestloss:
                    torch.save({
                        'pnet_state_dict': self.pnet.state_dict(),
                        'qnet_state_dict': self.qnet.state_dict(),
                    }, "best_model.pth")
                    self.bestloss = loss
                    self.pnet.train(True)
                    self.qnet.train(True)
                    
                    announce = 'New Best!'
                                
                fstring = f"{iter + 1:<10} {f'{vloss:.2e}':<10} {f'{e1:.2e}':<10} {f'{e2:.2e}':<15} {announce}"
                print(fstring)
                
        print('Best loss:', self.bestloss)
        
        checkpoint = torch.load("best_model.pth", weights_only=False)
        self.pnet.load_state_dict(checkpoint['pnet_state_dict'])
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.pnet.eval()
        self.qnet.eval()
