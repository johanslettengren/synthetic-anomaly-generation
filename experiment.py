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
    def __init__(self, N, djs, epanet_time, Q_true):
        self.N = N
        self.djs = djs
        self.epanet_time = epanet_time
        self.Q_true = Q_true
        self.Q_pred = None
        self.PINN_time = None
        
    def set_PINN_data(self, Q_pred, PINN_time):
        self.Q_pred = Q_pred
        self.PINN_time = PINN_time

def run_epanet(base_wn, Ns=[5], djs_list=[['2', '3', '6']], Cd=0.75, leak_ratio=0.3):


    results_list = []
    
    
    pipe_names = base_wn.pipe_name_list
    n_pipes = len(base_wn.pipe_name_list)
    
    diameters = np.array([pipe.diameter for _, pipe in base_wn.pipes()])
    
    leak_areas = np.pi * (diameters*leak_ratio / 2) ** 2
    
    
    for  N in Ns:
        
        for djs in djs_list:

            demand_idx = np.array([base_wn.node_name_list.index(dj) for dj in djs])

            demand_x = np.linspace(0, 1, N)
            demand_mesh = np.array(list(product(demand_x, repeat=len(demand_idx))))


            base_wn.options.time.duration = 0
            base_wn.options.quality.parameter = 'NONE'
            base_wn.options.hydraulic.demand_model = 'DDA'
            base_wn.options.hydraulic.emitter_exponent = 0.5

            data_collected = False

            if not data_collected:

                start = time.time()
                Q_true = np.zeros((demand_mesh.shape[0], n_pipes, n_pipes+1))

                # cache junction objects to avoid repeated lookups
                junction_objs = [base_wn.get_node(j) for j in djs]

                # --- pre-build one network per leak location (split once) ---
                leak_cases = []   # list of (pipe_name, wn_with_that_leak, leak_node_id)
                for i, p in enumerate(pipe_names):
                    wn_leak = copy.deepcopy(base_wn)
                    leak_node_id = f'LEAK-{i}'
                    # note: use unique virtual pipe id per split to avoid name clashes
                    wn_leak = wntr.morph.split_pipe(wn_leak, p, f'virt-{p}', leak_node_id)
                    wn_leak.get_node(leak_node_id).emitter_coefficient = Cd * leak_areas[i] * (2*g)**0.5
                    leak_cases.append((p, wn_leak, leak_node_id))

                # --- simulator choice: EpanetSimulator is faster for steady-state ---
                # Use WNTRSimulator if you rely on WNTR’s PDD/hydraulic features.
                Simulator = wntr.sim.EpanetSimulator  # or wntr.sim.WNTRSimulator

                n_demands = demand_mesh.shape[0]
                n_pipes = len(leak_cases)
                n_links = len(base_wn.link_name_list)

                for k, demand_vec in enumerate(demand_mesh):
                    # update base demands on every prebuilt leak network, run, collect
                    # (mutating in place is fine; we overwrite on the next iteration)
                    for i, (pipe_name, wn_leak, _) in enumerate(leak_cases):
                        # set the junction base demands for this scenario
                        for dj_val, j_name in zip(demand_vec, djs):
                            wn_leak.get_node(j_name).demand_timeseries_list[0].base_value = float(dj_val)

                        sim_results = Simulator(wn_leak).run_sim()
                        # steady-state => first (and only) timestamp
                        Q_true[k,i,:] = sim_results.link["flowrate"].iloc[0]#.reindex(wn_leak.link_name_list).to_numpy()
                    if (k+1) % 10 == 0 or k == n_demands - 1:
                        print(f"{k+1}/{n_demands}", end="\r", flush=True)
            end = time.time()
            results_list.append(Result(N, djs, end-start, Q_true))
            print(f'N = {N}, demand-junction-IDs={djs}, time={end-start}')
    return results_list



def run_PINN(base_wn, results, leak_ratio=0.3, print_interval=100):

    df = pd.DataFrame(columns=["N", "# demand junctions", "epanet time (s)", "PINN time", "mean NRMSD"])

    n_pipes = len(base_wn.pipe_name_list)
    diameters = np.array([pipe.diameter for _, pipe in base_wn.pipes()])
    leak_areas = np.pi * (diameters*leak_ratio / 2) ** 2
    
    for result in results:
        
        djs = result.djs
        
        model_params = get_model_parameters(base_wn)
        demand_idx = np.array([base_wn.node_name_list.index(dj) for dj in djs])
        model_params['demand_idx'] = demand_idx
        model_params['N'] = result.N
        model_params['a'] = torch.tensor(leak_areas, dtype=torch.float32)
        model_params['Q_true'] = result.Q_true

        net_params = {
                'layer_sizes' : [len(djs)+1,250,250,250,model_params['A0'].shape[1]], 
                'activation' : 'tanh',
        }

        start = time.time()

        demand_x = np.linspace(0, 1, result.N)
        demand_mesh = np.array(list(product(demand_x, repeat=len(djs))))

        D = torch.tensor(demand_mesh, dtype=torch.float32)
        ID = torch.arange(n_pipes).reshape(-1, 1)          
        D_exp  = D.unsqueeze(1).expand(-1, ID.shape[0], -1) 
        ID_exp = ID.unsqueeze(0).expand(D.shape[0], -1, -1)

        x = torch.cat([D_exp, ID_exp.float()], dim=-1)   

        model = Model(model_params, net_params)
        model.train(iterations=10000, print_interval=print_interval)
        Q = model.net(x)                         
        idx = (n_pipes + ID.unsqueeze(0).expand(Q.shape[0], -1, 1)).long() 
        Q_sel = Q.gather(dim=-1, index=idx)  
        Q_pred = torch.cat((Q[...,:Q.shape[-1]//2], Q_sel), dim=-1).detach().numpy() 

        end = time.time()
        Q_true = result.Q_true
        
        rng = Q_true.max(-1) - Q_true.min(-1)
        nrmsd = np.mean((Q_pred - Q_true)**2,-1) / rng 

        #result_list.append(nrmsd_range_summary(Q_pred,Q_trues[i]))
        
        df.loc[len(df)] = [result.N, len(djs), result.epanet_time, end-start, nrmsd.mean()]

        print(f'N = {result.N}, t={end-start}')
        
        result.set_PINN_data(Q_pred, end-start)

    return results, df