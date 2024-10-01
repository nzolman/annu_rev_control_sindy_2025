import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from sparse_rev.odes import lorenz

from sparse_rev.fit_utils import fit, pred_next_step
from sparse_rev.data_gen import data_gen_scale


T_max = 2
dt = 0.02
T_warmup = 1.0

# number of timesteps to wait for the dynamics to live on the attractor
N_warmup = int(T_warmup // dt) 


def prep_lorenz_data(sigma = 1e-2, 
                     signal_scales=np.array([8.16,  8.40, 26.94]), 
                     n_traj = 16, train_seed=0, T_max = T_max
                     ):
    '''
    Prepare the data for training/testing using a Lorenz dataset.
    
    Arguments:
        sigma (float)
            noise fraction to add to each scaled dimension
        signal_scales (list [x_scale, y_scale, z_scale])
            the x,y, and z scales for the data. Default values precomputed based on absoluate value of data from the attractor.
        n_traj (int)
            number of trajectories
        train_seed (int)
            seed for generating training data.
        T_max (float)
            time duration of the trajectory
    '''
    t = np.arange(0, T_warmup + T_max, dt)
    
    
    # format kwargs for data_gen_scale
    data_kwargs = dict(dt=dt, 
                        T_max=T_max+T_warmup, 
                        d=3,
                        gen='uniform',
                        sigmas = sigma* signal_scales, 
                        dyn_fn='lorenz', 
                        dyn_params = (10, 2.66667, 28)
                        )
    
    # grab training data
    key, data = data_gen_scale(seed=train_seed, n_traj=n_traj, 
                        gen_kwargs = dict(minval = -50, maxval = 50), 
                        **data_kwargs)
    
    # held out initial point near the attractor for test data
    x0 = np.array([2.5, 4, 12])
    test_data = solve_ivp(lorenz, y0=x0, 
                          t_span=[0,20], 
                          t_eval = np.arange(0,20,dt)
                          ).y.T.reshape(1,-1,3)
    
    return key, data, {'clean': test_data}


def get_rel_errors_trajs(model_name, 
                         n_traj_list, 
                         sigma = 1e-2,
                         signal_scales=np.array([8.16,  8.40, 26.94]),  seed = 0
                         ):
    '''
    Sweep through the n_traj_list to get the relative errors for a particular model class.
    
    Arguments:
        model_name (str)
            type of model to use (e.g. "sindy", "gp", "nn", etc.)
        n_traj_list (list)
            list of n_traj values to sweep over
        sigma (float)
            noise fraction
        signal_scales (list[x_scale, y_scale, z_scale])
            the x,y, and z scales for the data. Default values precomputed based on absoluate value of data from the attractor.
        seed (int)
            seed for generating training data.
    '''
    t = np.arange(0,T_max+T_warmup, dt)[N_warmup:]
    
    rel_errors = []
    models = []
    for n_traj in tqdm(n_traj_list): 
        
        key, data, test_data = prep_lorenz_data(sigma=sigma, 
                                                n_traj = n_traj, 
                                                train_seed=seed, 
                                                )
        
        # only return training dataset after the warmup (on the strange attractor)
        train_data = data['noise'][:,N_warmup:]
        
        noise_scale = np.sum(signal_scales) * sigma
        model = fit(model_name, train_data, sigma=noise_scale, t=t, dt=dt,seed=seed+2000)

        X0s = np.concatenate(test_data['clean'][:,:-1])
        X_hat = np.concatenate(test_data['clean'][:,1:])
        preds = pred_next_step(model_name, model, X0s, dt=dt, train_data = train_data)
        
        rel_err =np.mean((preds - X_hat)**2 /(1e-6 + np.linalg.norm(X_hat, axis=1)**2).reshape(-1,1)) 
        rel_errors.append(rel_err)
    
        models.append(model)
    return rel_errors, models


def get_rel_errors_noise(model_name, 
                         sigma_list, 
                         signal_scales=np.array([8.16,  8.40, 26.94]), n_traj = 16, 
                         seed = 0
                         ):
    '''
    Sweep through the sigma list to get the relative errors for a particular model class
    
    Arguments:
        model_name (str)
            type of model to use (e.g. "sindy", "gp", "nn", etc.)
        sigma_list (list)
            list of noise fractions to sweep over
        n_traj (int)
            number of trajectories
        signal_scales (list[x_scale, y_scale, z_scale])
            the x,y, and z scales for the data. Default values precomputed based on absoluate value of data from the attractor.
        seed (int)
            seed for generating training data.
    '''
    
    t = np.arange(0,T_max+T_warmup, dt)[N_warmup:]
    
    rel_errors = []
    models = []
    for sigma in tqdm(sigma_list): 
        
        key, data, test_data = prep_lorenz_data(sigma=sigma, 
                                                n_traj = n_traj, 
                                                train_seed=seed, 
                                                )
        train_data = data['noise'][:,N_warmup:]
        
        noise_scale = np.sum(signal_scales) * sigma
        model = fit(model_name, train_data, sigma=noise_scale, t=t, dt=dt,seed=seed+2000)

        X0s = np.concatenate(test_data['clean'][:,:-1])
        X_hat = np.concatenate(test_data['clean'][:,1:])
        
        preds = pred_next_step(model_name, model, X0s, dt=dt, train_data = train_data)
        
        rel_err = np.mean((preds - X_hat)**2 /(1e-6 + np.linalg.norm(X_hat, axis=1)**2).reshape(-1,1)) 
        rel_errors.append(rel_err)
        models.append(model)
    return rel_errors, models


def get_rel_errors_t_max(model_name, 
                         T_max_list,
                         sigma = 1.0,
                         n_traj = 16, 
                         signal_scales=np.array([8.16,  8.40, 26.94]), 
                         seed = 0):
    '''
    Sweep through the T_max_list to get the relative errors for a particular model class
    
    Arguments:
        model_name (str)
            type of model to use (e.g. "sindy", "gp", "nn", etc.)
        T_max_list (list)
            list of T_max values to sweep over
        sigma (float)
            noise fraction
        n_traj (int)
            number of trajectories
        signal_scales (list[x_scale, y_scale, z_scale])
            the x,y, and z scales for the data. Default values precomputed based on absoluate value of data from the attractor.
        seed (int)
            seed for generating training data.
    '''
    rel_errors = []
    models = []
    for T_max in tqdm(T_max_list): 
        t = np.arange(0,T_max+T_warmup, dt)[N_warmup:]
        key, data, test_data = prep_lorenz_data(sigma=sigma, 
                                                n_traj = n_traj, 
                                                train_seed=seed,
                                                T_max = T_max)
        train_data = data['noise'][:,N_warmup:]
        
        noise_scale = np.sum(signal_scales) * sigma
        model = fit(model_name, train_data, sigma=noise_scale, t=t, dt=dt,seed=seed+2000)

        X0s = np.concatenate(test_data['clean'][:,:-1])
        X_hat = np.concatenate(test_data['clean'][:,1:])
        
        preds = pred_next_step(model_name, model, X0s, dt=dt, train_data = train_data)
        
        rel_err = np.mean((preds - X_hat)**2 /(1e-6 + np.linalg.norm(X_hat, axis=1)**2).reshape(-1,1)) 
        rel_errors.append(rel_err)
        models.append(model)
    return rel_errors, models