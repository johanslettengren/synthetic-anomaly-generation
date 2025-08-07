import torch
import torch.nn as nn
from scipy.constants import g

softplus = nn.Softplus()
relu = nn.ReLU()


class Net(nn.Module):

    def __init__(self, layer_sizes, activation, base, U, mask=False, positive=False):
        super().__init__()
        
        self.U = U
        self.mask = mask
        self.positive = positive
        self.base = torch.tensor(base[None,:], dtype=torch.float32)
        self.activation = {'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 'sigmoid' : nn.Sigmoid(), 'softplus' : softplus}[activation] 
        
        
        layer_sizes = layer_sizes
        
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.linears.append(layer)
            
        
    def forward(self, x):
        
        for linear in self.linears[:-1]:

            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self.positive:
            x = softplus(x)
            
        if self.mask:
            idx = x[:,1].long()
            x = self.U.T[idx] * x
            
        return x
    

class Model():
    def __init__(self, model_params, net_params):  
        
        required_keys = ['A0', 'inv', 'M', 'B', 'a', 'S', 'D', 'L', 'd', 'Cd', 'C', 'rho']

        # Assert all required keys are present
        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        # Assign attributes
        for key in required_keys:
            setattr(self, key, model_params[key])
            
            
        self.n_samples = len(self.a)

        self.L = self.L[None,:]
        self.D = self.D[None,:]
        self.d = self.d[None,:]
        self.C = self.C[None,:]
        self.supply = (self.B @ self.S)[None,:]
        
        self.n_pipes = self.M.shape[1]
                    
        self.net = Net(**net_params)        
                
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.net.parameters(), lr=0.001)
        self.mse = lambda x : (x**2).mean()
        
        self.bestloss = 1e12
                
        self.mse = lambda x : (x**2).mean()

    def hL(self, q):
        return torch.sign(q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * self.L * torch.abs(q)**(1.852)
    
    def d_leak(self, a, H):
        d = self.Cd * a * torch.sqrt(2 * g * relu(H))
        return d
    
    def mv(self, M, v):
        return (M @ v.T).T

    
    def loss(self, D, idx):
        
        self.D[:,[1,3,5]] = D
        
        out = torch.clone(self.D)
    
        input = torch.cat((D, idx), dim=-1)
        out[:,(6.0+idx).long()] = self.net(input)
        
        q = self.mv(self.inv.T, out)
        
        hL = self.hL(q)    
        H = self.mv(self.inv, self.supply - hL)
        
        
        a_full = torch.zeros((self.n_samples, self.n_pipes))
        a_full[torch.arange(self.n_samples),idx.long()] = self.a
            
        
        
        loss = self.mse(self.mv(self.A0, q) - self.D - self.d_leak(self.mv(self.M, a_full), H))
        
        return loss 

            
    def train(self, iterations, print_interval=100):
        self.steps = []
        self.net.train(True)
        print(f"{'step':<10} {'loss':<10} {'e1':<10}  {'e2'}")
        
        for iter in range(iterations):
            
            
            idx = torch.randint(self.n_pipes, (1,1), dtype=torch.float32)
            D = torch.stack((torch.rand((1)), torch.rand((1)), torch.rand((1))), dim=-1)
                                                                        
            self.optimizer.zero_grad()
            loss = self.loss(D, idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            vloss = loss.item()
            
            announce = ''
            if iter % print_interval == print_interval - 1:   
                             
                if vloss < self.bestloss:
                    torch.save(self.net.state_dict(), "best_model.pth")    
                    self.bestloss = loss
                    
                    announce = 'New Best!'
                # {f'{e1:.2e}':<10} {f'{e2:.2e}':<15}                
                fstring = f"{iter + 1:<10} {f'{vloss:.2e}':<10} {announce}"
                print(fstring)
                
        
    def eval(self):
        print('Best loss:', self.bestloss)
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()

def loss_(self, a):
        
    H = self.net(a)
        
    hL = (self.B @ self.S)[None,:None] - self.mv(self.A0.T, H).squeeze(-1)
    

    q = (torch.sign(hL) * (torch.abs(hL) * self.C[None,:]**(1.852) * self.d[None,:]**(4.871) / 10.667 / self.L[None,:])**(1 / 1.852))
    loss = self.mse(self.mv(self.A0, q).squeeze(-1) - self.D[None,:] - self.d_leak((self.M @ a.T).T, H))
    
    return loss 

def loss_(self, a):
                    
    H, q = self.net(a)
    e1 = self.mse(self.mv(self.A0, q) - self.D - self.d_leak(a, H)) 
    e2 = self.mse(self.B @ self.S - self.mv(self.A0.T, H) - self.hL(q))
    
    return e1, e2

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