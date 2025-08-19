import wntr
import numpy as np
import networkx as nx
import torch
from scipy.sparse import dok_matrix



def get_model_parameters(wn):
    
    
    pipe_names = wn.pipe_name_list
    n_pipes = len(wn.pipe_name_list)
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
    
    
    #L_mat = A0 @ A0.T
    #L_chol = torch.linalg.cholesky(L_mat)
    
    model_params = {
        'A0': A0,
        'inv' : inv,
        'M' : M,
        'B' : B,
        #'a' : leak_areas,
        #'demand_idx' : demand_idx,     
        'S' : S,
        #'D' : D,
        'd': d,
        'L': L,
        'Cd' : 0.75, 
        'C' : C,
        'rho' : 1000.0,
        'n_samples' : n_pipes,
    }
    
    return model_params