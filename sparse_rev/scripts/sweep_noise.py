import os
import numpy as np
import pandas as pd
from copy import deepcopy

from sparse_rev.bench_utils import get_rel_errors_noise
from sparse_rev import _parent_dir

if __name__ == '__main__':
    _SAVE_DIR = os.path.join(_parent_dir, 'data', 'noise')
    N_SEEDS = 10
    N_TRAJ = 5
    SIGMA_LIST  = np.linspace(1e-2, 2e-1, 10)
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
            rel_errors, m = get_rel_errors_noise(model_name, SIGMA_LIST, n_traj = N_TRAJ, seed=seed)

            model_dicts = []
            for sigma_idx, sigma in enumerate(SIGMA_LIST):
                model_dicts.append(deepcopy({'model_name': model_name, 'sigma': sigma, 'err': rel_errors[sigma_idx]}))
            model_df = pd.DataFrame(model_dicts)
            save_path = os.path.join(_SAVE_DIR, f'{model_name}_{seed}.csv')
            model_df.to_csv(save_path, index=False)
            
        all_models.append(model_df)

    # export everything to a single csv
    save_path = os.path.join(_SAVE_DIR, f'all_noise.csv')
    all_df = pd.concat(all_models)
    all_df.to_csv(save_path, index=False)