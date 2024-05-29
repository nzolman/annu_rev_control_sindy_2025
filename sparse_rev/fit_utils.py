import numpy as np
from jax import vmap

from sparse_rev import _tmp_dir
from sparse_rev.sindy_utils import pareto_sindy, predict_next_steps_sindy
from sparse_rev.pysr_utils import pareto_pysr, predict_next_steps_pysr, simple_pysr_fit
from sparse_rev.dmd_utils import fit_dmd
from sparse_rev.nn_utils import train_nn
from sparse_rev.gp_utils import fit_ker, label_position, prep_data

def fit(model_name, train_data, sigma=1e-2, t=np.arange(0,1,0.1), dt=0.1):
    if model_name == 'sindy': 
        model = pareto_sindy(train_data, sigma=sigma, dt =dt, 
                                t = t, threshold_list = np.logspace(-5,0,10))
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
        model, _, _ = train_nn(train_data, n_batch = 1, n_epochs = int(1e4), lr=1e-3)
        
    elif model_name =='gp': 
        use_sparse_gp = (train_data.shape[0] * train_data.shape[1]) > 500
        model = fit_ker(train_data, sigma=sigma, use_sparse_gp=use_sparse_gp)
    else:
        raise KeyError(f'Invalid Name: {model_name}')
    return model
    
def pred_next_step(model_name, model, X, dt, n=1, train_data = None):
    if model_name == 'sindy': 
        preds = predict_next_steps_sindy(model,X,dt, n=n)
    elif model_name == 'pysr':
        preds =  predict_next_steps_pysr(model,X,dt, n=n)
        
    elif model_name == 'dmd':
        preds = np.array([model.simulate(x0, t = 1 + n)[-1] for x0 in X])
    
    elif model_name == 'nn':
        preds = X + vmap(model)(X)
    elif model_name == 'gp':
        d = X.shape[-1]
        dataset_train = prep_data(train_data)
        distr = model.predict(label_position(X.T, n_dim=d),
                            dataset_train)
        preds = distr.mean().reshape(-1,d)
    else:
        raise KeyError(f'Invalid Name: {model_name}')

    return preds