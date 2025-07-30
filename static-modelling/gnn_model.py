import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

softplus = nn.Softplus()

class GNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        self.edge_attr = edge_attr
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_attr = self.edge_attr
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(edge_input)

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))

class LeakGNN(nn.Module):
    def __init__(self, num_nodes, num_edges, pipe_attrs, node_dim=16, edge_dim=16, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.pipe_const = pipe_attrs  # shape [num_edges, attr_dim]
        self.node_embed = nn.Linear(1, node_dim)
        self.edge_embed = nn.Linear(pipe_attrs.shape[1], edge_dim)

        self.layers = nn.ModuleList([GNNLayer(node_dim, edge_dim) for _ in range(num_layers)])
        self.final_node = nn.Linear(node_dim, 1)  # head H
        self.final_edge = nn.Linear(edge_dim, 1)  # flow q

    def forward(self, leak_area, edge_index):
        x = self.node_embed(leak_area)                     # [num_nodes, node_dim]
        e = self.edge_embed(self.pipe_const)               # [num_edges, edge_dim]

        for layer in self.layers:
            x = layer(x, edge_index, e)

        H = self.final_node(x)                             # [num_nodes, 1]
        q = self.final_edge(e)                             # [num_edges, 1]
        return H.squeeze(-1), q.squeeze(-1)                # [N], [E]


class Model():
    def __init__(self, model_params, gnn_params):  
        required_keys = ['A0', 'B', 'A_max', 'S', 'D', 'L', 'd', 'Cd', 'C', 'rho', 'n_samples']

        missing = [key for key in required_keys if key not in model_params]
        if missing:
            raise KeyError(f"Missing required model parameters: {missing}")

        for key in required_keys:
            setattr(self, key, model_params[key])
            
        self.edge_index = self.edge_index_from_incidence(self.A0)
        
        # Precompute constant edge features: [C, d, L]
        edge_const = torch.stack([
            self.C.expand(self.A0.shape[1]), 
            self.d.expand(self.A0.shape[1]),
            self.L.expand(self.A0.shape[1])
        ], dim=1)

        self.gnn = LeakGNN(
            num_nodes=self.A0.shape[0],
            num_edges=self.A0.shape[1],
            pipe_attrs=edge_const,
            **gnn_params
        )
        
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.001)
        self.bestloss = 1e12
        
    def edge_index_from_incidence(self, A0):
        """
        Converts an incidence matrix A0 [num_nodes x num_edges]
        to edge_index [2 x num_edges] for PyTorch Geometric.
        """
        sources = []
        targets = []

        for j in range(A0.shape[1]):  # for each edge
            source = (A0[:, j] == -1).nonzero(as_tuple=True)[0]
            target = (A0[:, j] == 1).nonzero(as_tuple=True)[0]

            if len(source) == 1 and len(target) == 1:
                sources.append(source.item())
                targets.append(target.item())
            else:
                raise ValueError(f"Edge {j} must have one source and one target in A0.")

        return torch.tensor([sources, targets], dtype=torch.long)

    def hL(self, q):
        return torch.sign(q) * 10.667 * self.C**(-1.852) * self.d**(-4.871) * self.L * torch.abs(q)**(1.852)

    def d_leak(self, A, H):
        return self.Cd * A * torch.sqrt(2 * self.d * torch.abs(H))

    def loss(self, A):
        # A: [n_samples, 1]
        batch_size = A.shape[0]
        e1_total, e2_total = 0.0, 0.0

        for i in range(batch_size):
            A_i = A[i]                               # scalar
            leak = A_i * torch.ones((self.A0.shape[0], 1))  # [num_nodes, 1]

            H, q = self.gnn(leak, self.edge_index)

            mass_balance = self.A0 @ q - self.D - self.d_leak(leak.squeeze(), H)
            headloss = self.B @ self.S - self.A0.T @ H - self.hL(q)

            e1_total += (mass_balance ** 2).mean() / 0.1**2
            e2_total += (headloss ** 2).mean() / 50**2

        e1 = e1_total / batch_size
        e2 = e2_total / batch_size
        return e1, e2

    def train(self, iterations, print_interval=100):
        self.steps = []
        self.gnn.train(True)
        print(f"{'step':<10} {'loss':<10} {'e1':<10}  {'e2'}")

        for iter in range(iterations):
            A = self.A_max * torch.rand((self.n_samples, 1))  # batch of leak areas

            self.optimizer.zero_grad()
            e1, e2 = self.loss(A)
            loss = e1 + e2
            loss.backward()
            self.optimizer.step()

            if iter % print_interval == print_interval - 1:
                announce = ''
                if loss < self.bestloss:
                    torch.save(self.gnn.state_dict(), "best_gnn_model.pth")
                    self.bestloss = loss
                    announce = 'New Best!'
                
                print(f"{iter + 1:<10} {loss.item():.2e}   {e1.item():.2e}   {e2.item():.2e}   {announce}")

        print('Best loss:', self.bestloss)
        self.gnn.load_state_dict(torch.load("best_gnn_model.pth"))
        self.gnn.eval()
