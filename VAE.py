import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



class Encoder(nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation        
                
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        
        self.logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_normal_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

        
    def forward(self, x):        
        for linear in self.linears:
            x = self.activation(linear(x))
        
        return self.mu(x), self.logvar(x)
    
class Decoder(nn.Module):
    def __init__(self, layer_sizes, dx, activation, structure):
        super().__init__()
        
        self.structure = structure
        self.activation = activation    
        self.dx = dx    
                
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

    def forward(self, x):        
        
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
            
        x = self.linears[-1](x)
        
        if self.structure: 
            x = F.softplus(x) 
            x = x / ( torch.sum(x, dim=1, keepdim=True) * self.dx)
        return x
    
class VAE(nn.Module):
    def __init__(self, layer_sizes_enc, layer_sizes_dec, dx, activation, structure=True):
        super().__init__()
        
        self.layer_sizes_enc = layer_sizes_enc
        self.layer_sizes_dec = layer_sizes_dec
        self.activation_str = activation
        
        activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation] 
        
        self.encoder = Encoder(layer_sizes_enc, activation)
        self.decoder = Decoder(layer_sizes_dec, dx, activation, structure)
        

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar


class Model():

    def __init__(self, x, net):  
    
        self.x = x
        self.net = net        
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()        
    
    def loss(self, x_hat, mu, logvar):
        
        recon_loss = self.mse(x_hat - self.x)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()
            
    def train(self, iterations, val_interval=100):        
        self.net.train(True)
        print(f"{'step':<10} {'loss':<10} {'rec':<10} {'reg'}")
        for iter in range(iterations):            
                    
            self.optimizer.zero_grad()
            
            x_hat, mu, logvar = self.net(self.x) 
            loss, recon_loss, kl_loss = self.loss(x_hat, mu, logvar)
            loss.backward()
            self.optimizer.step()

            if iter % val_interval == val_interval - 1: 
                print(f"{iter + 1:<10} {f'{loss.item():.2e}':<10} {f'{recon_loss:.2e}':<10} {f'{kl_loss:.2e}'}")
                
        torch.save({
            'model_args': {
                'layer_sizes_enc': self.net.layer_sizes_enc,
                'layer_sizes_dec': self.net.layer_sizes_dec,
                'dx': self.net.decoder.dx,
                'activation': self.net.activation_str,
                'structure': self.net.decoder.structure
            },
            'state_dict': self.net.state_dict()
        }, 'VAE.pth')
        self.net.eval()
        


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