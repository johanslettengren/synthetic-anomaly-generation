import wntr
import networkx as nx
import torch

DEFAULT_CD = 0.75
DEFAULT_RHO = 1000.0


class ModelParameters:
    def __init__(self, wn : wntr.network.WaterNetworkModel):
        
        # List of named pipes in netowrk (before adding leak junctions)
        self.pipe_names = wn.pipe_name_list
        
        # Number of pipes in network
        self.n_pipes = len(wn.pipe_name_list)
        
        # List of names of (artificial) leak junctions
        self.leak_junctions = [f'LEAK-{i}' for i in range(self.n_pipes)]
        
        # Add (artificial) leak junctions - split pipes
        for i, pn in enumerate(self.pipe_names):
            wn = wntr.morph.split_pipe(wn, pn, pn + '_B', f'LEAK-{i}')
            
        self.wn = wn
        
        # List of junction names in network
        self.junction_names = self.wn.junction_name_list
        
        # Number of pipes in network
        self.n_pipes = len(wn.pipe_name_list)
        
        # List of reservoir names
        self.reservoirs = wn.reservoir_name_list 
                
        # List of edges in DiGraph
        self.edgelist = [(pipe.start_node_name, pipe.end_node_name) for _, pipe in wn.pipes()]
        
        self.G = self.build_DiGraph()
        self.M = self.build_M_matrix()
        self.B = self.build_B_matrix()
        self.S = self.build_S_vector()  
            
        # indices of non-reservoir junctions
        non_reservoir_idx = [i for i, n in enumerate(wn.node_name_list) if n not in self.reservoirs]

        # Incidence matrix
        A  = nx.incidence_matrix(self.G, nodelist=wn.node_name_list, edgelist=self.edgelist, oriented=True)
        
        # Reduced incidence matrix (no reservoir rows)
        self.A0 = torch.tensor(A[non_reservoir_idx,:].toarray(), dtype=torch.float32)
        
        # Get length of each pipe (in meters)
        self.L = torch.tensor([pipe.length for _, pipe in self.wn.pipes()], dtype=torch.float32)
     
        # Get diameter of each pipe (in meters)
        self.d = torch.tensor([pipe.diameter for _, pipe in self.wn.pipes()], dtype=torch.float32)
        
        # Get elevations
        self.elev = torch.tensor([wn.get_node(name).elevation for name in self.junction_names], dtype=torch.float32)
        
        # Get Hazen-Williams roughness coefficients (unitless)
        self.C = torch.tensor([pipe.roughness for _, pipe in self.wn.pipes()], dtype=torch.float32)

        # Pre-compute pseudo-inverse of reduced incidence matrix
        self.inv = torch.linalg.pinv(self.A0.T)
        

    def build_DiGraph(self) -> nx.DiGraph:
        """Create a clean DiGraph with no multiple edges"""
        G = nx.DiGraph()
        for pipe_name, pipe in self.wn.pipes():
            G.add_edge(pipe.start_node_name, pipe.end_node_name, name=pipe_name)
        
        return G

    def build_M_matrix(self) -> torch.tensor:
        """Create mapping matrix M (local leak node index to global leak node index)"""
        
        node_to_index = {n: i for i, n in enumerate(self.junction_names)}
        M = torch.zeros((len(self.junction_names), len(self.leak_junctions)), dtype=torch.float32)
        for j, leak_junction in enumerate(self.leak_junctions):
            M[node_to_index[leak_junction], j] = 1.0
            
        return M

    def build_B_matrix(self) -> torch.tensor:
        """Build matrix B (local supply node idx to global supply pipe idx)"""
        
        B = torch.zeros((len(self.edgelist), len(self.reservoirs)), dtype=torch.float32)
        for i, edge in enumerate(self.edgelist):
            if edge[0] in self.reservoirs:
                B[i, self.reservoirs.index(edge[0])] = 1.0
        return B
    
    def build_S_vector(self) -> torch.tensor:
        """Get vector of heads at reservoirs"""
        S_values = []
        for name in self.reservoirs:
            reservoir = self.wn.get_node(name)
            head = reservoir.base_head  # This is constant in steady state
            S_values.append(head)
        S = torch.tensor(S_values, dtype=torch.float32)
                
        return S
    

def get_model_parameters(wn, Cd=DEFAULT_CD, rho=DEFAULT_RHO):
    mp = ModelParameters(wn)
    return {
        'A0': mp.A0, 'inv': mp.inv, 'M': mp.M, 'B': mp.B, 'S': mp.S,
        'd': mp.d, 'L': mp.L, 'elev': mp.elev, 'Cd': Cd, 'C': mp.C, 'rho': rho,
        'edgelist': mp.edgelist, 'junctions': mp.junction_names, 'reservoirs': mp.reservoirs,
    }