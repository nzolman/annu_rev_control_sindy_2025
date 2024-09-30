import numpy as np
from jax import vmap

from sparse_rev import _tmp_dir
from sparse_rev.methods import  fit_dmd, pareto_sindy, predict_next_steps_sindy, \
                                predict_next_steps_pysr, simple_pysr_fit, \
                                train_nn, fit_ker, label_position, prep_data

def fit(model_name, train_data, sigma=1e-2, t=np.arange(0,1,0.1), dt=0.1, seed=0):
    '''
    Wrapper for fitting a given model class
    
    Arugments:
        model_name (str):
            which model to use. Must be in ['sindy', 'weak_sindy', 'dmd', 'pysr', 'nn', 'gp']
        train_data (ndarray): 
            training data
        sigma (float):
            noise parameter, generally used for defining L2 regularization
        t (ndarray):
            time mesh
        dt (float):
            timestep delta
        seed (int):
            random seed used for numpy and jax.
    '''
    np.random.seed(seed)
    
    if model_name == 'sindy': 
        model = pareto_sindy(train_data, sigma=sigma, dt =dt, 
                                t = t, threshold_list = np.logspace(-5,0,10), 
                                use_weak=False, use_ensemble=False)
    elif model_name == 'weak_sindy': 
        
        # note that for WSINDy, we don't have the same L2 Regularization because the "noise" in
        # induced by the integration of the library, not by the direct measurment noise itself. Try fixing
        # this parameter. 
        w_sigma = 1e-3
        
        # simple heuristic for defining how many basis functions to use
        K= max((int(t.max()/dt) + 1), 1000)
        dK = t.max() / K
        Hxt = 4 * max(dt, (dK + dt)/2)
        
        model = pareto_sindy(train_data, sigma=w_sigma, dt =dt, t = t, 
                             threshold_list = np.logspace(-5,0,10), 
                             use_weak=True, weak_kwargs = {'K': K, 'p':5, 'H_xt': Hxt},
                             use_ensemble=True
                            )

    elif model_name == 'dmd':
        model = fit_dmd(train_data, sigma = sigma, dt = dt, t=t)

    elif model_name == 'pysr':
        # in practice, the parsimony parameter is nearly negligble here. To save time,
        # we can just set this to its default fixed number
        
        # p_space = np.logspace(-5,1,10)
        # model = pareto_pysr(train_data, dt =dt, t = t, p_space = p_space, 
        #                     tmp_dir = _tmp_dir
        #                     )
        model = simple_pysr_fit(train_data, t = t, tmp_dir = _tmp_dir)
        
    elif model_name == 'nn': 
        # take the best model before overfitting
        (_, _, _), (model, _, _) = train_nn(train_data, n_batch = 1, n_epochs = int(1e4),
                                            lr=1e-4, seed=seed)
        
    elif model_name =='gp': 
        # save significant amount of time using sparse GP for larger datasets
        use_sparse_gp = True
        n_traj_val = max(int(len(train_data) * 0.2), 1)
        model, _ = fit_ker(train_data, sigma=sigma, use_sparse_gp=use_sparse_gp, seed=seed, 
                           use_diff=False, n_traj_val = n_traj_val, max_val=100)
    elif model_name =='gp_diff': 
        n_traj_val = max(int(len(train_data) * 0.2), 1)
        use_sparse_gp = True
        model, _ = fit_ker(train_data, sigma=sigma, use_sparse_gp=use_sparse_gp, seed=seed, use_diff=True, 
                        n_traj_val = n_traj_val, max_val=100)
    else:
        raise KeyError(f'Invalid Name: {model_name}')
    return model

    
def pred_next_step(model_name, model, X, dt, n=1, train_data = None):
    '''Wrapper function for predicting next step for a given model class
    
    Arguments:
        - model_name (str)
            model class
        - model (object)
            model to use
        - X (ndarray)
            array of initial conditions
        - dt (float)
            timedelta between steps
        - n (int)
            number of steps forward
        - train_data (ndarray)
            training dataset. Only used for GP to setup the kernel.
    '''
    if model_name in ['sindy', 'weak_sindy']: 
        preds = predict_next_steps_sindy(model,X,dt, n=n)

    elif model_name == 'pysr':
        preds =  predict_next_steps_pysr(model,X,dt, n=n)
        
    elif model_name == 'dmd':
        preds = np.array([model.simulate(x0, t = 1 + n)[-n] for x0 in X])
    
    elif model_name == 'nn':
        preds = X + vmap(model)(X)

    elif model_name == 'gp':
        d = X.shape[-1]
        dataset_train = prep_data(train_data, use_diff=False)
        distr = model.predict(label_position(X.T, n_dim=d),
                            dataset_train)
        preds = distr.mean().reshape(-1,d)

    elif model_name == 'gp_diff':
        d = X.shape[-1]
        dataset_train = prep_data(train_data, use_diff=True)
        distr = model.predict(label_position(X.T, n_dim=d),
                            dataset_train)
        preds = distr.mean().reshape(-1,d) + X

    else:
        raise KeyError(f'Invalid Name: {model_name}')

    return preds