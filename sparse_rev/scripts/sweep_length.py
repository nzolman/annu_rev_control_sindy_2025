import os
import numpy as np
import pandas as pd
from copy import deepcopy

from sparse_rev.bench_utils import get_rel_errors_t_max
from sparse_rev import _parent_dir

if __name__ == '__main__':
    _SAVE_DIR = os.path.join(_parent_dir, 'data', 'tmax')
    N_SEEDS = 10
    N_TRAJ = 5
    SIGMA = 0.01 # use 1% noise
    TMAX_LIST = np.logspace(-1, 7, 10, base=2) # number of seconds per trajectory
    
    os.makedirs(_SAVE_DIR, exist_ok=True)
    
    model_names = [
                'sindy', 
                'pysr', 
                'nn', 
                'dmd', 
                'gp',
                'weak_sindy'
                ]

    all_models = []
    for seed in range(N_SEEDS):
        for model_name in model_names:
            print(model_name)
            rel_errors, m = get_rel_errors_t_max(model_name, 
                                                 T_max_list=TMAX_LIST, 
                                                 sigma=SIGMA, 
                                                 n_traj=N_TRAJ, 
                                                 seed=seed)

            model_dicts = []
            for tmax_idx, tmax in enumerate(TMAX_LIST):
                model_dicts.append(deepcopy({'model_name': model_name, 'tmax': tmax, 'err': rel_errors[tmax_idx]}))
            model_df = pd.DataFrame(model_dicts)
            save_path = os.path.join(_SAVE_DIR, f'{model_name}_{seed}.csv')
            model_df.to_csv(save_path, index=False)
            
            all_models.append(model_df)

    # export everything to a single csv
    save_path = os.path.join(_SAVE_DIR, f'all_tmax.csv')
    all_df = pd.concat(all_models)
    all_df.to_csv(save_path, index=False)