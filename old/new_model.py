import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def hf(V):
    return 10.67 * 1200 * V**1.852 / 100**1.852

def grad(y, x):
    o = torch.ones_like(y[...,0])
    g = torch.cat([torch.autograd.grad(y[...,i], x, grad_outputs=o, create_graph=True)[0] for i in range(y.shape[-1])],dim=-1)
    return g

class branch(nn.Module):
    """Postive Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation
        self.networks = nn.ModuleList()
        
        layer_sizes = layer_sizes
                
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
    
    def forward(self, u):
        
        for linear in self.linears[:-1]:
            u = self.activation(linear(u))
        return self.linears[-1](u)

         

class trunk(nn.Module):
    
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
        layer_sizes = layer_sizes
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def forward(self, x):
        
        o = torch.zeros_like(x)
        for linear in self.linears[:-1]:
            o = self.activation(linear(o))
        
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
            
        o = self.linears[-1](o)
        x = self.linears[-1](x)
                
        return x - o
    

class Model(nn.Module):
    def __init__(self, layer_sizes_trunk, layer_sizes_branch, activation, n_nets=1):
        super().__init__()

        activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh()}[activation] 
        
        self.e = 0
        
        self.n_nets = n_nets
        self.trunk = nn.ModuleList(trunk(layer_sizes_trunk, activation) for _ in range(n_nets))
        self.branch = nn.ModuleList(branch(layer_sizes_branch, activation) for _ in range(n_nets))
        
        self.mse = lambda x : (x**2).mean()
        
    def forward(self, x, u):
        return 50 + torch.stack([torch.einsum("uk,xk->ux", self.branch[i](u), self.trunk[i](x)) for i in range(self.n_nets)])
        
    def loss(self, x, u):
        
        t = [t(x) for t in self.trunk]
        
        x_end = 1200 * torch.ones_like(x)
        t_end = [t(x_end) for t in self.trunk]         
        
        b = [b(u) for b in self.branch]
        
        t_x = [grad(s,x) for s in t]
        
        
        dNdx = torch.stack([torch.einsum("uk,xk->ux", b[i], t_x[i]) for i in range(self.n_nets)])   
        
        N_end = torch.stack([torch.einsum("uk,xk->ux", b[i], t_end[i]) for i in range(self.n_nets)])  
        
        # if self.e == 0:
        #     dNdx = dNdx.detach()
            
        # else: 
        #     #N_end = N_end.detach()
        #     dNdx = dNdx.detach()
            
        # self.e += 1
        
        return self.mse(dNdx + hf(0.05 + N_end.detach())) * 1e11
        
        
    
class Trainer():
    def __init__(self, model, lr=0.001):  
        
        self.x_coll = torch.linspace(0, 1200, 200, requires_grad=True).reshape(-1, 1)
        self.u_coll = 0.001*torch.ones(100, 1)  #torch.linspace(0, 0.001, 10, requires_grad=True).reshape(-1, 1)
        
        
        self.model = model
    
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.mse = lambda x : (x**2).mean()
        
        self.losshistory = []
        self.bestvloss = 1000000

    

    
            
    def train(self, iterations, val_interval=100):
        
                
        self.steps = []
        self.model.train(True)
        print(f"{'step':<10} {'loss':<10}")
        
        for iter in range(iterations):
                
            loss = self.model.loss(self.x_coll, self.u_coll)

            loss.backward()
            self.optimizer.step()

            if iter % val_interval == val_interval - 1:
                self.model.eval()            
                loss = loss.item()
                
                announcement = ''
                if loss < self.bestvloss:
                    announcement = 'New best model!'
                    torch.save(self.model.state_dict(), "best_model.pth")    
                    self.bestvloss = loss                  
                    self.losshistory.append(loss)
                    self.steps.append(iter)
                    self.model.train(True)
                                
                fstring = f"{iter + 1:<10} {f'{loss:.2e}':<10} {announcement}"
                print(fstring)

                
        self.model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.model.eval()
        
        
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