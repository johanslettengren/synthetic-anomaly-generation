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
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
        
    def forward(self, x):

        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)

        if self.positive:
            x = softplus(x)
        return x
    

class Model():
    def __init__(self, model_params, net_params):  
        
        required_keys = ['A0', 'B', 'A_max', 'S', 'D', 'L', 'd', 'Cd', 'C', 'rho', 'n_samples']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
                    
        self.Hnet = Net(self.A0.shape[0], **net_params, positive=True)
        self.qnet = Net(self.A0.shape[1], **net_params)
        
        params = itertools.chain(self.Hnet.parameters(), self.qnet.parameters())

                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(params, lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.bestloss = 1e12
                
        self.mse = lambda x : (x**2).mean()

    def hL(self, q):
        return torch.sign(q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * self.L * torch.abs(q**(1.852))
    
    def d_leak(self, A, H):
        return self.Cd * A * torch.sqrt(2 * self.d * torch.abs(H))
    
    def mv(self, M, v):
        v_ = v.reshape(-1, v.shape[-1])
        return (M @ v_.T).T.reshape(v.shape[0], v.shape[1], -1)

    def loss(self, A):
                
        H = self.Hnet(A)
        q = self.qnet(A)
        
                
        e1 = self.mse(self.mv(self.A0, q) - self.D - self.d_leak(A, H))
        e2 = self.mse(self.B @ self.S - self.mv(self.A0.T, H) - self.hL(q))
        
        
        return e1, e2

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.qnet.train(True)
        self.Hnet.train(True)
        print(f"{'step':<10} {'loss':<10} {'e1':<10}  {'e2'}")
        
        for iter in range(iterations):
            A = self.A_max * torch.rand((self.n_samples, 1))
                                    
            self.optimizer.zero_grad()
            e1, e2 = self.loss(A)
            loss = e1 + e2
                
            vloss = e1 + e2
            loss.backward()
            self.optimizer.step()
            
            announce = ''
            if iter % print_interval == print_interval - 1:           
                if vloss < self.bestloss:
                    torch.save({
                        'Hnet_state_dict': self.Hnet.state_dict(),
                        'qnet_state_dict': self.qnet.state_dict(),
                    }, "best_model.pth")
                    self.bestloss = loss
                    self.Hnet.train(True)
                    self.qnet.train(True)
                    
                    announce = 'New Best!'
                                
                fstring = f"{iter + 1:<10} {f'{vloss:.2e}':<10} {f'{e1:.2e}':<10} {f'{e2:.2e}':<15} {announce}"
                print(fstring)
                
        print('Best loss:', self.bestloss)
        
        checkpoint = torch.load("best_model.pth", weights_only=False)
        self.Hnet.load_state_dict(checkpoint['Hnet_state_dict'])
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.Hnet.eval()
        self.qnet.eval()
