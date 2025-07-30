import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import itertools


from numpy import pi
from scipy.constants import g

softplus = nn.Softplus()


class Net(nn.Module):

    def __init__(self, dim_in, layer_sizes, activation, positive=False):
        super().__init__()
        
        self.positive = positive
        self.activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid()}[activation] 
        self.networks = nn.ModuleList()
        
        
        layer_sizes = [dim_in] + layer_sizes
        
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
    
class GNN(nn.Module):

    def __init__(self, A0, D, B, S, depth, layer_sizes, activation):
        super().__init__()
        
        self.D = D
        self.A0 = A0
        self.B = B
        self.S = S
        self.depth = depth

        self.node_net = Net(2, layer_sizes, activation)
        self.edge_net = Net(1, layer_sizes, activation)
        
        self.q = nn.Parameter(torch.randn(self.A0.shape[1]))

        
    def mv(self, M, v):
        v_ = v.reshape(-1, v.shape[-1])
        return (M @ v_.T).T.reshape(v.shape[0], v.shape[1], -1)
    
    def forward(self, H):
        
        # H.shape = (n_samples, n_nodes)

        q = self.q
        
        m1 = self.A0 @ q - self.D
        m1 = m1.expand(H.shape)

        x = torch.stack((H, m1), dim=-1)
        
        # x.shape = (n_samples, n_nodes, 2)
        
        H = H.unsqueeze(-1)
        q = torch.zeros(H.shape[0], self.A0.shape[1], 1)
        for _ in range(self.depth):
        
            H = H + self.node_net(x)
            # H.shape = (n_samples, n_nodes, 1)

            mat1 = self.B @ self.S
            # mat1.shape = n_edges
            
            mat2 = self.mv(self.A0.T, H)
            # mat2.shape = (n_samples, n_edges, 1)
            m2 = mat1[None,:,None] - mat2
            
            # m2.shape = (n_samples, n_edges, 1)
            q = q + self.edge_net(m2)
            # q.shape = (n_samples, n_edges, 1)
            
            
            m1 = self.mv(self.A0, q.squeeze(-1)) - self.D[None,:,None]
            # m1.shape = (n_samples, n_nodes, 1)
            x = torch.cat((H, m1), dim=-1)
            # x.shape = (n_samples, n_nodes, 2)
            
            

        return H.squeeze(-1), q.squeeze(-1)
    

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
                    
        self.net = GNN(self.A0, self.D, self.B, self.S, **net_params)
    
        
        #params = itertools.chain(self.Hnet.parameters(), self.qnet.parameters())
        
                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.bestloss = 1e12
                
        self.mse = lambda x : (x**2).mean()

    def hL(self, q):
        return torch.sign(q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * self.L * torch.abs(q)**(1.852)
    
    def d_leak(self, A, H):
        return self.Cd * A * torch.sqrt(2 * self.d * torch.abs(H) + 1e-3)
    
    def mv(self, M, v):
        v_ = v.reshape(-1, v.shape[-1])
        return (M @ v_.T).T.reshape(v.shape[0], v.shape[1], -1)

    def loss(self, a):
                
        H, q = self.net(a)
    
            
                
        e1 = self.mse(self.mv(self.A0, q) - self.D - self.d_leak(a, H)) 
        e2 = self.mse(self.B @ self.S - self.mv(self.A0.T, H) - self.hL(q))
        
        return e1, e2

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.net.train(True)
        print(f"{'step':<10} {'loss':<10} {'e1':<10}  {'e2'}")
        
        for iter in range(iterations):
            A = torch.linspace(0, self.A_max, self.n_samples).reshape(-1, 1)
            
                                    
            self.optimizer.zero_grad()
            e1, e2 = self.loss(A)
            loss = (e1 / (e1.detach() + e2.detach())) * e1 + (e2 / (e1.detach() + e2.detach())) * e2
                
            vloss = e1 + e2
            loss.backward()
            self.optimizer.step()
            
            announce = ''
            if iter % print_interval == print_interval - 1:           
                if vloss < self.bestloss:
                    #torch.save(self.net.state_dict(), "best_model.pth")    
                    self.bestloss = loss
                    
                    announce = 'New Best!'
                                
                fstring = f"{iter + 1:<10} {f'{vloss:.2e}':<10} {f'{e1:.2e}':<10} {f'{e2:.2e}':<15} {announce}"
                print(fstring)
                
        #print('Best loss:', self.bestloss)
        
        #self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        #self.net.eval()
