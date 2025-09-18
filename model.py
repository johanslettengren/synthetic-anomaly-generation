import torch
import torch.nn as nn
from scipy.constants import g
import numpy as np
from itertools import product

softplus = nn.Softplus()
relu = nn.ReLU()


class Net(nn.Module):
    """Simple neural network class"""
    def __init__(self, 
                 layer_sizes : list, 
                 activation : str, 
                 positive : bool = False):
        
        super().__init__()
        
        # Whether to produce positive output
        self.positive = positive
        
        # Set activate function
        self.activation = {'relu' : relu, 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid(), 'softplus' : softplus}[activation] 
                        
        # Create neural network layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.linears.append(layer)
            
        
    def forward(self, x):
        """Neural network forward pass"""
            
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        
        # Produce positive output
        if self.positive:
            x = softplus(x)
        return  x
    

class Model(nn.Module):
    """Full PINN model"""
    
    def __init__(self, model_params, net_params):  
        super().__init__()
        
        
        required_keys = ['A0', 'inv', 'M', 'B', 'a', 'S', 'demand_idx', 'L', 'elev', 'd', 'Cd', 'C', 'rho', 'N']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
            
        # Modify network variables 
        self.L = 2*self.L[None,:self.L.shape[0] // 2]
        self.elev = self.elev[None,:]
        self.d = self.d[None,:]
        self.C = self.C[None,:]
        self.supply = (self.B @ self.S)[None,:]
        self.n_pipes = self.M.shape[1]
        self.n_samples = model_params['n_samples']
        self.leak_id = torch.arange(self.n_pipes, dtype=torch.float32).reshape(-1,1)
        
        # Create demand grid
        demand_x = torch.linspace(0, 1, self.N, dtype=torch.float32)
        self.D = torch.tensor(list(product(demand_x, repeat=len(self.demand_idx))), dtype=torch.float32)
        self.D = self.D.reshape(self.D.shape[0], 1, 1, 1, -1)
        
        
        # Build full input grid of scenario parameters (for validation)
        self.ID = torch.arange(self.n_pipes).reshape(-1, 1)
        ID = self.ID.reshape(1, 1, 1, -1, 1)         
        sr_ref = torch.tensor(model_params['split ratios ref'], dtype=torch.float32).reshape(1,-1, 1, 1, 1)    
        lr_ref = torch.tensor(model_params['leak ratios ref'], dtype=torch.float32).reshape(1, 1, -1, 1, 1)
        D_exp  = self.D.expand(-1, sr_ref.shape[1], lr_ref.shape[2], ID.shape[3], self.D.shape[-1]) 
        sr_ref_exp = sr_ref.expand(self.D.shape[0], -1, lr_ref.shape[2], ID.shape[3], sr_ref.shape[-1]) 
        lr_ref_exp = lr_ref.expand(self.D.shape[0], sr_ref.shape[1], -1, ID.shape[3], lr_ref.shape[-1]) 
        ID_exp = ID.expand(self.D.shape[0], sr_ref.shape[1], lr_ref.shape[2], -1, ID.shape[-1])
        self.x_val = torch.cat([D_exp, sr_ref_exp, lr_ref_exp, ID_exp.float()], dim=-1)    
        
        # Get reference solution       
        self.Q_true = model_params['Q_true']
        
        # Get full layer sizes
        layer_sizes = net_params['layer_sizes']
        layer_sizes = [len(self.demand_idx)+3] + layer_sizes + [self.A0.shape[1]]
        
        self.scenario_params = net_params['scenario params']    
        
        # Build neural network           
        self.net = Net(layer_sizes, net_params['activation'])        
                
        # Define optimizer
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.net.parameters(), lr=0.001)
        
        # Define base loss
        self.mse = lambda x : (x**2).mean()
        
        # Pre-define best loss to track progress
        self.bestloss = 1e12
                
    def hL(self, Q, L):
        """Calculate head loss due to friction"""
        return torch.sign(Q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * L * torch.abs(Q)**(1.852)

    def d_leak(self, a, H):
        """Calculate output flow due to leaks"""
        return self.Cd * a * torch.sqrt(2 * g * relu(H - self.elev))
    
    def mv(self, M, v):
        """Batched matrix-vector multiplication"""
        return (M @ v.T).T
    
    def prod(self, x, y):
        """Batched cartesian product"""
        x_exp  = x.unsqueeze(1).expand(-1, y.shape[0], -1) 
        y_exp = y.unsqueeze(0).expand(x.shape[0], -1, -1)
        prod = torch.cat([x_exp, y_exp], dim=-1)
        return prod.reshape(-1, prod.shape[-1]) 
    
    def loss(self, D, sr, lr):
        """Physics informed loss function"""
                
        # Get all pipe lengths (of split pipes)
        # Based on original pipe lengths and split ratios
        L = torch.cat((self.L * sr, self.L * (1-sr)), dim=-1)
                
        # Get batch input grid
        idx_exp = self.leak_id.repeat(self.n_samples, 1)
        input = torch.cat((D, sr, lr, idx_exp), dim=-1)
        idx = idx_exp.long()        

        # Insert demand at relevant demand junction into full demand vector -- batched
        demand = torch.zeros(D.shape[0], self.A0.shape[0])
        demand[:,self.demand_idx] = D
              
        # Get diameter of relevant index - batched
        diameters = self.d[0,idx]
        
        # Get corresponding leak areas
        a = np.pi * (diameters*lr / 2) ** 2
        
        # Insert leak areas at relevant leak junctions into full leak-area vector -- batched
        areas = torch.zeros((idx_exp.shape[0], self.n_pipes))
        areas.scatter_(1, idx, a) # self.a[idx]

        # Generate prediction
        Q = self.net(input)
        
        # Calculate head loss due to friction
        hL = self.hL(Q, L)  
        
        # Calculate hydraulic heads corresponding to flows
        H = self.mv(self.inv, self.supply - hL) 
        
        # Compute physics loss 'for Q'
        loss1 = self.mse(self.mv(self.A0, Q) - demand - self.d_leak(self.mv(self.M, areas), H))
        
        # Compute physics loss 'for H'
        loss2 = self.mse(self.supply - hL - self.mv(self.A0.T, H))
        
        return loss1, loss2

            
    def train(self, print_interval=100, threshold=0.9):
        """Train neural network - optimize w.r.t physics loss"""
        
        self.steps = []
        self.net.train(True)
        
        p = 0.0
        iter = 0
        
        # Criterion: p10 < threshold (in nrmsd)
        while p < threshold:
            
            D = torch.rand((self.n_samples*self.n_pipes, self.demand_idx.shape[0]))
            
            # Generate demand, split-ratio and leak-ratio vectors 
            # split-ratio / leak-ratio not considered -> default values
            if 'split ratios' in self.scenario_params:
                sr = torch.rand((self.n_samples*self.n_pipes, 1))
            else:
                sr = torch.tensor([0.5], dtype=torch.float32).repeat(self.n_samples*self.n_pipes,1)
            if 'leak ratios' in self.scenario_params:
                lr = 0.3*torch.rand((self.n_samples*self.n_pipes, 1))
            else:
                lr = torch.tensor([0.3], dtype=torch.float32).repeat(self.n_samples*self.n_pipes,1)
            
            # One step of gradient descent (over batch)                               
            self.optimizer.zero_grad()
            loss1, loss2 = self.loss(D, sr, lr)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()
            
            # Check criterion (and print current status)
            # Not at every iteration due to computational cost         
            if iter % print_interval == print_interval - 1:  
                _, nrmse = self.validate() 
                p = float(np.mean(nrmse.reshape(-1) <= 0.1))
                print(f'p10={100*p:.2f}%, mean nrmsd={nrmse.mean()}', end="\r", flush=True)
            iter += 1
            
    def validate(self):
        """Validate current model -- check prediction against reference solution"""
        
        self.net.eval()
        with torch.no_grad():  
            
            # Generate solution over full grid of scenario parameters
            Q = self.net(self.x_val)  

            # PINN predictios has 2 x n_pipes & EpaNet only has n_pipes + 1
            # We pick out only the relevant split pipe in the PINN prediction
            # The rest of the predictions is set to the first of the two pipe halves (they should have the same flow)
            idx = (self.n_pipes + self.ID.reshape(1, 1, 1, -1, 1).expand(Q.shape[0], Q.shape[1], Q.shape[2], -1, 1)).long() 
            Q_sel = Q.gather(dim=-1, index=idx)  
            Q_pred = torch.cat((Q[...,:Q.shape[-1]//2], Q_sel), dim=-1).detach().numpy() 
            
            # Calculate nrmsd -- batched
            rmse = np.sqrt(((Q_pred - self.Q_true) ** 2).mean(axis=-1))
            rng = self.Q_true.max(axis=-1) - self.Q_true.min(axis=-1)
            
            nrmse = rmse / rng

            return Q_pred, nrmse
        
       