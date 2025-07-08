import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NN(nn.Module):
    """Vanilla Neural Network

    Args:
        layer_sizes : shape of network
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        self.activation = activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation]   
                
        
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    def forward(self, z):
        """NN forward pass"""

        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)

        return z

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
    def __init__(self, x_train, y_train, x_test, y_test, net, lr=0.001, val_interval=100):  
        
        # Training data
        self.x_train = x_train
        self.y_train = y_train
        
        # Testing data
        self.x_test = x_test
        self.y_test = y_test
                
        # For saving the best validation loss
        self.bestvloss = 1000000
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation loss

        # Initialize Adam optimizer
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=lr)
        
        # Set MSE loss function
        self.loss = torch.nn.MSELoss()
        
        # Number of iterations between validations
        self.val_interval = val_interval
        
    # def format(self, x, requires_grad=False):
    #     """Convert data to torch.tensor format with data type float32"""
    #     x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    #     return x.to(torch.float32).requires_grad_(requires_grad)
        
    def train(self, iterations):
        """Train network"""
        
        # Train step history
        self.steps = []
        
        # Set net to training mode
        self.net.train(True)
        
        # For displaying losses upon validation
        print('Step \t Train loss \t Test loss')
        
        for iter in range(iterations):
            """Training iteration"""    
                    
            # Set gradients to zero
            self.optimizer.zero_grad()
            
            # Get network ouput for training data
            outputs = self.net(*self.x_train)            
            
            # Calculate corresponding loss  
            loss = self.loss(outputs, self.y_train)
            
            # Calculate gradients
            loss.backward()
            
            # Gradient descent
            self.optimizer.step()

            if iter % self.val_interval == self.val_interval - 1:
                """Validation of network"""
                
                # Set network to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    
                    # Get network output for validation data
                    outputs = self.net(*self.x_test)  
                    
                    # Calculate corresponding loss
                    vloss = self.loss(outputs, self.y_test)
                    
                    # Check if we have a new best validation loss
                    announce_new_best = ''
                    if vloss < self.bestvloss:
                        
                        # If we do, announce this
                        announce_new_best = 'New best model!'
                        
                        # Save current mode
                        torch.save(self.net.state_dict(), "best_model.pth")    
                        
                        # Update current best validation loss
                        self.bestvloss = vloss               
                        
        
                    # Save loss history
                    tloss = loss.item() # training loss
                    self.vlosshistory.append(vloss)
                    self.tlosshistory.append(tloss)
                    self.steps.append(iter)
                    
                    # Set net to training mode again
                    self.net.train(True)
                
                # Display losses at vaildation iteration
                print('{} \t [{:.2e}] \t [{:.2e}] \t {}'.format(iter + 1, tloss, vloss, announce_new_best))    
                
        # Load the model with best validation loss
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        
        # Set network to evalutation mode (training done)
        self.net.eval()
        
        
    def plot_losshistory(self, dpi=100):
        # Plot the loss trajectory
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.plot(self.steps, self.tlosshistory, '.-', label='Training loss')
        ax.plot(self.steps, self.vlosshistory, '.-', label='Test loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()   