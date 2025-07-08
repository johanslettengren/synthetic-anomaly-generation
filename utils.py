import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


# # Physical parameters
beta = 1 #2.2e9 
A = 1        
rho = 1         
F = 1            
L = 1        
Nz = 200
h = L / Nz

NX = 10

g = 9.81

# Time span
t0, tmax = 0, 10

Nt = 50
t_eval = np.linspace(t0, tmax, Nt)

phi = lambda z : torch.zeros_like(z)

def solve(q0, pl, d, X):
    
    # initial conditions
    def initial_conditions():
        p_init = np.ones(Nz) * pl(0)
        q_init = np.ones(Nz) * q0(0)
        return np.concatenate([q_init, p_init])

    # RHS of ODE system
    def pq_rhs(t, y):
        
        q = y[:Nz]
        p = y[Nz:]
        
        q = np.append(q0(t), q)
        p = np.append(p, pl(t))
        
        p_rhs = - (beta / A) * (q[1:] - q[:-1]) / h - (beta / A) * d * X        
        q_rhs = - (A / rho) * (p[1:] - p[:-1]) / h - (F / rho) * q[1:] - (1 / A) * d * X**2

        return np.concatenate([q_rhs, p_rhs])

    # Solve the system
    y0 = initial_conditions()

    return solve_ivp(pq_rhs, [t0, tmax], y0, t_eval=t_eval, method='RK45')

class NN(nn.Module):
    """Postive Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
                
        layer_sizes = [1] + layer_sizes + [1]
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    def forward(self, x):
        
        """NN forward pass"""
        
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
            
        x = self.linears[-1](x)
        #x = self.softplus(x)

        return x


class branch(nn.Module):
    """Postive Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
        self.K = K
        
        self.networks = nn.ModuleList()
        
        layer_sizes = [1] + layer_sizes + [2*K]
        
                
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
        self.networks.append(self.linears)
            
        
    def forward(self, X):
        
        """NN forward pass"""
             
        bX = X
        for linear in self.linears[:-1]:
            bX = self.activation(linear(bX))
        
        bX = self.linears[-1](bX).view(-1, self.K, 2)
        
        # b0 = torch.zeros_like(X, dtype=torch.float32)
        # for linear in self.linears[:-1]:
        #     b0 = self.activation(linear(b0))
        
        # b0 = self.linears[-1](b0).view(-1, self.K, 2)
                   
        return bX
    
class trunk(nn.Module):
    """Postive Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
        self.softplus = nn.Softplus()
        
        layer_sizes = [2] + layer_sizes + [K]
        
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def fwd(self, zt):
        for linear in self.linears[:-1]:
            zt = self.activation(linear(zt))
        return self.linears[-1](zt)
        
        
        
    def forward(self, zt):
        
        t = zt[:,1][:Nt]
        z = zt[:,0].view(-1, Nt)[:, 0]
        
        n = int(zt.shape[0]/Nt)
        _t = torch.cartesian_prod(torch.zeros(n, dtype=torch.float32), t)
        z_ = torch.cartesian_prod(z, torch.zeros(Nt, dtype=torch.float32))
        zeros = torch.zeros_like(zt)
        
        azt = self.fwd(zt)
        a0t = self.fwd(_t)
        az0 = self.fwd(z_)
        a00 = self.fwd(zeros)
                    
        return azt - az0 - a0t + a00

class DeepONet(nn.Module):
    """Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes_trunk, layer_sizes_branch, K, activation):
        super().__init__()


        activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation] 
        
        self.K = K
        self.trunk = trunk(layer_sizes_trunk, K, activation)
        self.branch = branch(layer_sizes_branch, K, activation)
        
    def forward(self, zt, X):
        """DeepONet forward pass"""
                
        n = int(zt.shape[0]/Nt)
        trunk_output = self.trunk(zt)
        branch_output = self.branch(X)
        output = torch.einsum("ekd,xk->xed", branch_output, trunk_output)

        return output.view(n, Nt, NX, 2)
    
    # def integral(self, zt, X):
        
    #     trunk_output  = self.trunk(zt)
        
    #     branch_output = self.branch(X)
        
    #     Itrunk = trunk_output.view(z.shape[0], t.shape[0], -1).mean(dim=0) / L

    #     ddt_list = [torch.autograd.grad(Itrunk[...,i], t, grad_outputs=torch.ones_like(Itrunk[...,0]), create_graph=True)[0] \
    #                 for i in range(self.K)]
    #     ddt = torch.stack(ddt_list, dim=-1)
        
    #     dI_dt = torch.einsum("ekd,xk->xed", branch_output, ddt)
    #     I = torch.einsum("Xkd,xk->xed", branch_output, Itrunk)

    #     return dI_dt, I
    
    def derivative(self, zt, X):
        
        trunk_output  = self.trunk(zt)
        branch_output = self.branch(X)
        
        dadx = torch.stack(
            [
            torch.autograd.grad(
                trunk_output[..., i],
                zt,
                grad_outputs=torch.ones_like(trunk_output[..., 0]),
                create_graph=True
            )[0]
            for i in range(self.K)
            ],
            dim=-1
        )
        
        n = int(zt.shape[0]/Nt)
        
        dadz, dadt = dadx[:,0,:], dadx[:,1,:]  
        
        output = torch.einsum("ekd,xk->xed", branch_output, trunk_output).view(n, Nt, NX, 2)
        ddz = torch.einsum("ekd,xk->xed", branch_output, dadz).view(n, Nt, NX, 2)
        ddt = torch.einsum("ekd,xk->xed", branch_output, dadt).view(n, Nt, NX, 2)
        
        q_z = ddz[...,0]
        p_z = -ddz[...,1].flip(dims=(0,))
        
        q_t = ddt[...,0]
        p_t = ddt[...,1].flip(dims=(0,))
        
        q = output[...,0]

        return q_z, p_z, q_t, p_t, q


        
    

class Model():
    """Model for Training Networks

    Args:
        x_train (tuple) : input training data
        y_train : target training data
        x_test (tuple) : input validation data
        y_test : target validation data
        net : network to train
        lr : learning rate of optimiser
        val_interval : number of iterations between validations
    """
    def __init__(self, y, net, d):  
        
        self.z = torch.linspace(0, L, Nz, dtype=torch.float32)
        self.t = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True)
        self.zt = torch.cartesian_prod(self.z, self.t)
        
        
        
        self.y = y  # training (q0, ql, p0, pl) 
        
        self.X = torch.linspace(0, 1, NX, dtype=torch.float32)[None,:]
        
        
        
        
        self.d = torch.tensor(d, dtype=torch.float32)[...,None]
        
        # For saving the best validation loss
        self.bestvloss = 1000000
        
        # Network
        self.net = net        

        
        # Loss history
        self.losshistory = []  # training loss

        # Initialize Adam optimizer
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=0.001)
        
        # Set MSE loss function
        self.mse = lambda x : (x**2).mean()
                
        self.loss_weight = 1
        
    # def format(self, x, requires_grad=False):
    #     """Convert data to torch.tensor format with data type float32"""
    #     x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    #     return x.to(torch.float32).requires_grad_(requires_grad)
    
    def data_loss(self, output, y):
        
        fixed_loss = self.mse(y[0,:,None] - output[0,:,:,0]) + self.mse(y[3,:,None] - output[1,:,:,1])
        
        #perturbed_loss = self.mse(y[1,:,None] - output[1,:,:,0]) + self.mse(y[2,:,None] - output[0,:,:,1]) # might change to incorporate sign
        
        return fixed_loss #- perturbed_loss
    
    def sanity_check():
        pass
    
    def Iloss(self):
        
        dI_dt, I = self.net.integral(self.z_coll, self.t_coll, self.X.T)   
        ends = self.net(torch.tensor([0, L], dtype=torch.float32), self.zt[:,1][:Nt], self.X.T) 
                
        term1 = self.mse(A * dI_dt[...,1] + beta * (ends[1,...,0] - ends[0,...,0] + self.X)) 
        
        term2 = self.mse(rho * A * dI_dt[...,0] + A * A * (ends[0,...,1] - ends[1,...,1]) + F * A * I[...,0] + rho * self.X * self.X)
                        
        ploss = term1 + term2
        
        return ploss, ends
    
    def dloss(self):
        
        q_z, p_z, q_t, p_t, q = self.net.derivative(self.zt, self.X.T)      
        
        ends_zt = torch.cartesian_prod(torch.tensor([0, L], dtype=torch.float32), self.zt[:,1][:Nt])
        
        
        ends = self.net(ends_zt, self.X.T) 

        X = self.X[None,...]
        
        
        r1 = self.mse(p_t + (beta / A) * q_z + (beta / A) * self.d * X)
        r2 = self.mse(q_t + (A / rho) * p_z + (F / rho) * q + (1 / A) * self.d * X**2) # +  A * g * sin_theta(z) 
                 
        # r1_rel = r1 / (p_t.abs().mean()**2 + q_z.abs().mean()**2)
        # r2_rel = r2 / (q_t.abs().mean()**2 + p_z.abs().mean()**2 + q.abs().mean()**2)
        
        return r1, r2, ends
    
            
    def train(self, iterations, val_interval=100):
        """Train network"""
        
        # Train step history
        self.steps = []
        
        # Set net to training mode
        self.net.train(True)
        
        # For displaying losses upon validation
        print(f"{'step':<12} {'loss':<12} {'r1':<12}  {'r2':<21} {'dql':<27} {'dp0':<25} {'ql':<26} {'p0':<23}")
        
        for iter in range(iterations):
            """Training iteration"""    
            
                    
            # Set gradients to zero
            self.optimizer.zero_grad()
            
            # Get network ouput for training data
            #output = self.net(*self.x_data, X)                
            
            # Calculate corresponding loss  
            #dloss = self.data_loss(output, self.y)
            r1, r2, ends = self.dloss()
            
            loss = r1 + r2
                                    
            # Calculate gradients
            loss.backward()
            
            # Gradient descent
            self.optimizer.step()

            if iter % val_interval == val_interval - 1:
                """Validation of network"""
                
                # Set network to evalutation mode
                self.net.eval()            
                loss = loss.item()
                                
                dql, dp0 = ends[1,...,0], ends[1,...,1]
                
                new = self.y + ends
                
                dql_min, dql_max = dql.min(), dql.max()
                dp0_min, dp0_max = dp0.min(), dp0.max()
                
                
                ql, p0 = new[1,...,0], new[1,...,1]
                ql_min, ql_max = ql.min(), ql.max()
                p0_min, p0_max = p0.min(), p0.max()
            
                
                # Check if we have a new best validation loss
                announce_new_best = ''
                if loss < self.bestvloss:
                    
                    # If we do, announce this
                    announce_new_best = 'New best model!'
                    
                    # Save current mode
                    torch.save(self.net.state_dict(), "best_model.pth")    
                    
                    # Update current best validation loss
                    self.bestvloss = loss                  
        
                    # Save loss history
                    self.losshistory.append(loss)
                    self.steps.append(iter)
                
                   # Set net to training mode again
                    self.net.train(True)
                
                # Display losses at vaildation iteration
                
                fstring = (
                    f"{iter + 1:<10} {f'{r1+r2:.2e}':<10} {f'{r1:.2e}':<10} {f'{r2:.2e}':<15}"
                    f"{f'[{dql_min:.1e}, {dql_max:.1e}]':<27}"
                    f"{f'[{dp0_min:.1e}, {dp0_max:.1e}]':<27}"
                    f"{f'[{ql_min:.1e}, {ql_max:.1e}]':<27}"
                    f"{f'[{p0_min:.1e}, {p0_max:.1e}]':<25}"
                    f"{announce_new_best}"
                )
                
                print(fstring)
                #print(f"{'':<26} {f'[{rql_min:.1e}, {rql_max:.1e}]':<30} [{rp0_min:.1e}, {rp0_max:.1e}]")   
                
        # Load the model with best validation loss
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        
        # Set network to evalutation mode (training done)
        self.net.eval()
        
        
    def plot_losshistory(self, dpi=100):
        # Plot the loss trajectory
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.plot(self.steps, self.losshistory, '.-', label='Loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()   
        
        

def generate_gaussian_leaks(num_leaks):

    z = np.linspace(0, 1, Nz).reshape(-1, 1)

    #kernel = Matern(length_scale=0.1, nu=2.0)
    kernel = RBF(length_scale=0.1)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    
    samples = gp.sample_y(z, n_samples=num_leaks, random_state=0) 
    
    samples = samples - np.percentile(samples, 85, axis=0, keepdims=True)        

    samples = (samples + np.abs(samples)) / 2
    samples = samples / np.trapezoid(samples, z, axis=0)
    
    # plt.plot(z[:,0], samples[:,0], linewidth=2.5)
    # plt.fill_between(z[:,0], samples[:,0], alpha=0.2)
    # plt.show()
    
    #samples = np.zeros_like(samples)
    
    return samples