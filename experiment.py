import wntr
import numpy as np
import torch
import copy
from itertools import product
import time
from scipy.constants import g
import pandas as pd

from model import *
from config import *




class Result:
    def __init__(self, N : int, 
                 djs : list, sr : list, 
                 lr : list, 
                 epanet_time : float, 
                 Q_true : np.array):
        
        # List of numbers of grid points
        self.N = N
        
        # List of demand junctions
        self.djs = djs
        
        # Corresponding simulation time for EpaNet
        self.epanet_time = epanet_time
        
        # Generated reference solutions
        self.Q_true = Q_true
        
        # List of split ratios
        self.sr = sr
        
        # List of leak ratios
        self.lr = lr
        
    def set_PINN_data(self, Q_pred, PINN_time):
        
        # PINN predictionion
        self.Q_pred = Q_pred
        
        # PINN train + prediticiton time
        self.PINN_time = PINN_time

def run_epanet(base_wn, N_list=[5], split_ratios=[0.5], leak_ratios=[0.3], djs_list=[['2', '3', '6']], Cd=0.75):
    """Runs the experiment for EpaNet simulator"""
    
    # List to store Result objects
    results_list = []
    
    leak_ratios = np.array(leak_ratios)
    pipe_names = base_wn.pipe_name_list
    n_pipes = len(base_wn.pipe_name_list)
    
    # Get pipe diamaeters
    diameters = np.array([pipe.diameter for _, pipe in base_wn.pipes()])[None,:]
    
    # Transform leak ratios to leak areas (depending on pipe diamaters)
    leak_areas = np.pi * (diameters*leak_ratios[:,None] / 2) ** 2    
    
    # Iteratite trhough the leak scenarios
    for N in N_list:
        for djs in djs_list:

            # Get indices of demand nodes
            demand_idx = np.array([base_wn.node_name_list.index(dj) for dj in djs])
            
            # Get demand mesh
            demand_x = np.linspace(0, 1, N)
            demand_mesh = np.array(list(product(demand_x, repeat=len(demand_idx))))

            # Set simulation options
            base_wn.options.time.duration = 0
            base_wn.options.quality.parameter = 'NONE'
            base_wn.options.hydraulic.demand_model = 'DDA'
            base_wn.options.hydraulic.emitter_exponent = 0.5
            
            n_splits = len(split_ratios)
            n_areas = len(leak_ratios)

            start = time.time()
            Q_true = np.zeros((demand_mesh.shape[0], n_splits, n_areas, n_pipes, n_pipes+1))

            leak_cases = []
                
            # Prebuild networks for leak scenarios
            for split_ratio in split_ratios:
                fixed_area = []
                for leak_area in leak_areas:
                    fixed_pipe = []
                    for id, p in enumerate(pipe_names):
                        
                        # Copy network to add leak
                        wn_leak = copy.deepcopy(base_wn)
                        leak_node_id = f'LEAK-{id}'
                        
                        # Add (virtual) leak node
                        wn_leak = wntr.morph.split_pipe(wn_leak, p, f'virt-{p}', leak_node_id, split_at_point=split_ratio)
                        
                        # Add leak to leak node
                        wn_leak.get_node(leak_node_id).emitter_coefficient = Cd * leak_area[id] * (2*g)**0.5
                        fixed_pipe.append(wn_leak)
                    fixed_area.append(fixed_pipe)
                leak_cases.append(fixed_area)
            
            # Create simulator
            Simulator = wntr.sim.EpanetSimulator 

            n_demands = demand_mesh.shape[0]

            # Run simulations
            for i, demand_vec in enumerate(demand_mesh):
                # update base demands on every prebuilt leak network, run, collect
                for j, fixed_area in enumerate(leak_cases):
                    for k, fixed_pipe in enumerate(fixed_area):
                        for l, wn_leak in enumerate(fixed_pipe):
                            # set the junction base demands for this scenario
                            for dj_val, j_name in zip(demand_vec, djs):
                                wn_leak.get_node(j_name).demand_timeseries_list[0].base_value = float(dj_val)

                            sim_results = Simulator(wn_leak).run_sim()
                            # steady-state => first (and only) timestamp
                            Q_true[i,j,k,l,:] = sim_results.link["flowrate"].iloc[0]
                            if (i+1) % 10 == 0 or j == n_demands - 1:
                                print(f"{i+1}/{n_demands}", end="\r", flush=True)
        end = time.time()
        results_list.append(Result(N, djs, split_ratios, leak_ratios, end-start, Q_true))
        print(f'N = {N}, demand-junction-IDs={djs}, time={end-start}')
    return results_list



def run_PINN(base_wn : wntr.network.WaterNetworkModel, 
             results : list, 
             scenario_params : list =[], 
             print_interval : int =100, 
             n_samples : int =10, 
             layer_sizes : list =3*[64], 
             threshold : float =0.9) -> tuple[Result, pd.DataFrame]:

    # Table for presenting results
    df = pd.DataFrame(columns=["N", "# demand junctions", "epanet time (s)", "PINN time", "mean NRMSD"])

    # Pipe diameters
    diameters = np.array([pipe.diameter for _, pipe in base_wn.pipes()])

    for result in results:
        
        start = time.time()
        
        djs = result.djs
        
        # Get network specific model parameters
        model_params = get_model_parameters(base_wn)
        demand_idx = np.array([base_wn.node_name_list.index(dj) for dj in djs])
        
        # Add scenario specific model parameters
        model_params['n_samples'] = n_samples
        model_params['demand_idx'] = demand_idx
        model_params['N'] = result.N
        model_params['split ratios ref'] = result.sr
        model_params['leak ratios ref'] = result.lr
                
        leak_areas = np.pi * (diameters*result.lr[:,None] / 2) ** 2
        
        model_params['a'] = torch.tensor(leak_areas, dtype=torch.float32)
        model_params['Q_true'] = result.Q_true

        # Specify network parameters
        net_params = {
                'layer_sizes' : layer_sizes,
                'activation' : 'tanh',
                'scenario params' : scenario_params
        }

        # Build model
        model = Model(model_params, net_params)
        
        # Train model
        model.train(print_interval=print_interval, threshold=threshold)
        
        # Produce predictions and calulate nmrsd
        Q_pred, nmrsd = model.validate()  

        end = time.time()
        
        # Add experiment data to table
        df.loc[len(df)] = [result.N, len(djs), result.epanet_time, end-start, nmrsd.mean()]

        # Print final time for this scenario
        print(f'N = {result.N}, t={end-start}')
        
        # Add results to result object
        result.set_PINN_data(Q_pred, end-start)

    return results, df