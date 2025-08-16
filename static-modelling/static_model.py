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
        required_keys = ['A0', 'inv', 'M', 'B', 'a', 'S', 'demand_idx', 'L', 'd', 'Cd', 'C', 'rho']

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
        
        self.n_samples = min([self.n_pipes, model_params['n_samples']])
                    
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
    
    def loss_(self,leak_id):
        
        demand = torch.zeros(self.n_samples, self.A0.shape[0])
        demand[:,self.demand_idx] = self.D.repeat(self.n_samples, 1)
        
        idx = leak_id.long()        
        batch_idx = torch.arange(self.n_samples).unsqueeze(1).expand_as(idx)
        
        areas = torch.zeros((self.n_samples, self.n_pipes))
        areas[batch_idx,idx] = self.a 
        
        H = self.net(leak_id)
        hL = self.supply - self.mv(self.A0.T, H)
    
        Q = self.hL_inv(hL)  

        loss = self.mse(self.mv(self.A0, Q) - demand - self.d_leak(self.mv(self.M, areas), H))

        return loss
    
    
    def loss(self, D, leak_id):
        
        demand = torch.zeros(self.n_samples, self.A0.shape[0])
        demand[:,self.demand_idx] = D
        
        idx = leak_id.long()        
        batch_idx = torch.arange(self.n_samples).unsqueeze(1).expand_as(idx)
        
        areas = torch.zeros((self.n_samples, self.n_pipes))
        areas[batch_idx,idx] = self.a 

    
        input = torch.cat((D, leak_id), dim=-1)
        
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
        print(f"{'step':<10} {'Q-loss':<10} {'H-loss':<10} {'sum-loss':<10}")
        
        for iter in range(iterations):
            
            
            idx = torch.arange(self.n_samples, dtype=torch.float32).reshape(-1,1)
            D = self.net.Dmax * torch.rand((self.n_samples, self.demand_idx.shape[0]))
            
                                                                        
            self.optimizer.zero_grad()
            loss1, loss2 = self.loss(D, idx)
            loss = loss1 + loss2
            loss.backward()
            #all_grads = torch.cat([p.grad.view(-1) for p in self.net.parameters() if p.grad is not None])
            #print(all_grads)
            self.optimizer.step()
            
            vloss = loss.item()
            
            announce = ''
            if iter % print_interval == print_interval - 1:   
                             
                if vloss < self.bestloss:
                    torch.save(self.net.state_dict(), "best_model.pth")    
                    self.bestloss = loss
                    
                    announce = 'New Best!'
                fstring = f"{iter + 1:<10} {f'{loss1.item():.2e}':<10} {f'{loss2.item():.2e}':<10} {f'{vloss:.2e}':<10} {announce}"
                print(fstring)
                
        
    def eval(self):
        print('Best loss:', self.bestloss)
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        
        
def get_model_parameters(wn):
    
    
    pipe_names = wn.pipe_name_list
    n_pipes = len(wn.pipe_name_list)
    non_leak_junction_names = wn.junction_name_list
    leak_junctions = [f'LEAK-{i}' for i in range(n_pipes)]
    
    for i, pn in enumerate(pipe_names):
        wn = wntr.morph.split_pipe(wn, pn, pn + '_B', f'LEAK-{i}')
    
    # Create a clean DiGraph with no multiple edges
    G = nx.DiGraph()

    # Rebuild the graph using pipe start → end as in WNTR
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        G.add_edge(pipe.start_node_name, pipe.end_node_name, name=pipe_name)
        
    # Build re-ordering of graph edges
    edges = list(G.edges())
    edge_order = []

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        idx = edges.index((pipe.start_node_name, pipe.end_node_name))
        edge_order.append(idx)
        
    edgelist = np.array(edges)[edge_order]
    
    # Build re-ordering of graph nodes
    nodes = list(G.nodes())
    node_order = []

    for node in wn.node_name_list:
        idx = nodes.index(node)
        node_order.append(idx)
        
    reservoirs = list(wn.reservoir_name_list)

    idx = [i for i, n in enumerate(wn.node_name_list) if n not in reservoirs]
    A  = nx.incidence_matrix(G, oriented=True)[:,edge_order][node_order,:]

    A0 = torch.tensor(A[idx,:].toarray(), dtype=torch.float32)
    
    
    # Create mapping matrix M (local leak node idx -> global leak node idx)
    junction_names = wn.junction_name_list
    # Map node names to row indices in your reduced incidence matrix A0
    node_to_index = {n: i for i, n in enumerate(junction_names)}

    M = torch.zeros((len(junction_names), len(leak_junctions)))

    # Set 1 where the leak area should be applied
    for j, leak_junction in enumerate(leak_junctions):
        i = node_to_index[leak_junction]
        M[i, j] = 1.0
        
    # Create mapping matrix B (local supply node idx -> global supply pipe idx)

    supply_nodes = wn.reservoir_name_list 
    supply_nodes = list(supply_nodes) 

    # Create edge-to-start-node mapping
    edge_start_nodes = [edge[0] for edge in edgelist]

    # Create B matrix (|E| x |supply_nodes|), sparse
    B = dok_matrix((len(edgelist), len(supply_nodes)), dtype=int)

    for i, (start_node) in enumerate(edge_start_nodes):
        if start_node in supply_nodes:
            j = supply_nodes.index(start_node)
            B[i, j] = 1

    # Convert to CSR format for efficient arithmetic
    B = torch.tensor(B.toarray(), dtype=torch.float32)
    
    # D_np = np.array(results.node['demand'].iloc[0][non_leak_junction_names].values, dtype=np.float32)
    # demand_idx = D_np.nonzero()[0]
    
    # D = torch.tensor(D_np[demand_idx], dtype=torch.float32)
    
    
    
    supply_nodes = wn.reservoir_name_list  # Or include tanks if needed

    # Get heads at each supply node
    S_values = []
    for name in supply_nodes:
        reservoir = wn.get_node(name)
        head = reservoir.base_head  # This is constant in steady state
        S_values.append(head)

    S = torch.tensor(S_values, dtype=torch.float32)
    
    pipe_names = wn.pipe_name_list 

    # Get length of each pipe (in meters)
    L = torch.tensor([wn.get_link(name).length for name in pipe_names], dtype=torch.float32)

    # Get diameter of each pipe (in meters)
    d = torch.tensor([wn.get_link(name).diameter for name in pipe_names], dtype=torch.float32)

    # Get Hazen-Williams roughness coefficients (unitless)
    C = torch.tensor([wn.get_link(name).roughness for name in pipe_names], dtype=torch.float32)

    inv = torch.linalg.pinv(A0.T)
    
    diameter = 1
    leak_ratio = np.array([0.3])
    leak_areas = 3.14159 * (diameter*leak_ratio / 2) ** 2
    
    
    #L_mat = A0 @ A0.T
    #L_chol = torch.linalg.cholesky(L_mat)
    
    model_params = {
        'A0': A0,
        'inv' : inv,
        'M' : M,
        'B' : B,
        'a' : torch.tensor(leak_areas, dtype=torch.float32),
        #'demand_idx' : demand_idx,     
        'S' : S,
        #'D' : D,
        'd': d,
        'L': L,
        'Cd' : 0.75, 
        'C' : C,
        'rho' : 1000.0,
        'n_samples' : 10
    }
    
    return model_params