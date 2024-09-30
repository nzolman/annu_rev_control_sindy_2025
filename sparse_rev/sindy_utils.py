import pysindy as ps
import numpy as np

from sklearn.model_selection import  LeaveOneOut, ShuffleSplit
from kneed import KneeLocator 


def fit_SINDy_model(train_data, alpha = 1e-2, threshold = 1e-1, dt = 0.01, cubic = True,
                    t_data=np.arange(0,0.1, 1e-2),
                    use_ensemble=False,
                    use_weak=True, t_grid=None, weak_kwargs = {},):
    '''
    Wrapper for fitting SINDy, E-SINDy, Weak SINDy, and Weak E-SINDy models
    '''
    
    # best practice for Weak libraries require is to explicitly define library functions
    library_functions = [lambda x: x, lambda x,y: x * y, lambda x,y,z: x*y*z]
    library_function_names = [lambda x: f'{x}', lambda x,y: f'{x}*{y}', lambda x,y, z: f'{x}*{y}*{z}']
    
    # can restrict to quadratic by removing the last terms
    if not cubic:
        library_functions.pop()
        library_function_names.pop()
    if t_grid is None: 
        t_grid =  t_data
        
    # use simple STRidge regression
    optimizer = ps.STLSQ(alpha = alpha, threshold=threshold)
    
    if use_ensemble:
        optimizer=ps.EnsembleOptimizer(optimizer, 
                                       bagging=True, 
                                       library_ensemble=True
                                       )

    # init and fit model
    if use_weak: 
        weak_lib = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            interaction_only=False,
            spatiotemporal_grid=t_grid,
            is_uniform=True,
            **weak_kwargs
            )
        weak_model = ps.SINDy(feature_library=weak_lib, optimizer=optimizer, t_default=dt)
        weak_model.fit(list(train_data), t=t_data, multiple_trajectories=True, quiet=True)
        model = weak_model

    else:
        lib = ps.CustomLibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            interaction_only=False
                            )
        model = ps.SINDy(feature_library=lib, optimizer=optimizer, t_default=dt)
        model.fit(list(train_data), t=t_data, multiple_trajectories=True, quiet=True)
    
    # for E-SINDy, set median coefs
    if use_ensemble:
        model.optimizer.coef_ = np.median(model.optimizer.coef_list, axis=0)
    return model
    

def cv_stlq_sindy(train_data, alpha = 1e-2, threshold = 1e-1, dt = 0.1, t=np.arange(0,20,0.1), 
                  n_splits=10, seed=0, use_weak=True, use_ensemble=False, weak_kwargs={}):
    '''assume train data has at least 2 trajs.'''
    alpha = max(alpha, 1e-6)
    n_dim = train_data.shape[-1]

    if (n_splits is str and n_splits == 'loo') or (n_splits < len(train_data)):
        loo = LeaveOneOut()
        splits = loo.split(train_data) 
    else:
        n_splits = min(len(train_data), n_splits)
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=.2, random_state=seed)
        splits = shuffle_split.split(train_data)
    
    mse_losses = []
    l0_losses = []
    for train_idx, val_idx in splits:
        # define model with given optimizer kwargs
        
        
        # model = ps.SINDy(feature_library=lib, optimizer=optimizer, t_default=dt)
        # model.fit(list(train_data[train_idx]), t=dt, multiple_trajectories=True, quiet=True)

        model = fit_SINDy_model(train_data[train_idx], 
                                alpha = alpha, 
                                threshold = threshold, 
                                dt = dt, 
                                t_data=t, 
                                use_weak=use_weak, 
                                use_ensemble=use_ensemble, 
                                weak_kwargs=weak_kwargs
                                )
        
        # predict on validations set
        # note, for the weak model, this prediction is NOT the velocities, it's the weak form. 
        val_pred = np.array([model.predict(train_data[idx]) for idx in val_idx])
        
        # figure out the "true derivatives"
        x_val, xdot_val = model._process_multiple_trajectories(train_data[val_idx], t, None)
        
        # reshape data
        val_pred = val_pred.reshape(-1, n_dim)
        xdot_val = np.array(xdot_val).reshape(-1,n_dim)
        
        # calculate losses
        mse_loss = float(np.mean((val_pred - xdot_val)**2))
        l0_loss = np.count_nonzero(model.optimizer.coef_)
        
        mse_losses.append(mse_loss)
        l0_losses.append(l0_loss)
        
    return np.array([mse_losses, l0_losses]).T


def pareto_sindy(train_data, sigma=1e-2, dt =0.1, 
                 t = np.arange(0,1,10), threshold_list = np.logspace(-5,0,10), 
                 use_weak = True, weak_kwargs = {},
                 use_ensemble=False
                 ):
    sigma = max(sigma, 1e-6)
    
    # perform cross validation on each of the prospective thresholds
    res = np.array([cv_stlq_sindy(train_data, threshold=thresh, alpha = sigma, dt = dt, t=t, 
                                  use_weak = use_weak, use_ensemble=use_ensemble, weak_kwargs=weak_kwargs)
                    for thresh in threshold_list])
    
    # find the pareto-optimal solution between accuracy and sparsity
    mses, l0s = res.mean(axis=1).T
    kneedle = KneeLocator( l0s,mses, S=1.0, curve="convex", direction='decreasing', online=True)
    
    if kneedle.elbow is None:
        elbow_idxes = np.array([[np.argmin(mses)]])
    else:
        elbow_idxes = np.where(l0s == kneedle.elbow)[0] 
    best_mse_idx = np.argmin(mses[elbow_idxes])

    best_elbow_idx = elbow_idxes[best_mse_idx]
    
    thresh = threshold_list[best_elbow_idx]
    
    # fit the model with the new value
    # NOTE: if use_weak==True, this fits a dummy model, we'll replace the coefficients with the real ones.
    model = fit_SINDy_model(train_data, alpha = sigma, threshold = thresh, dt = dt, t_data=t, 
                            use_weak=False, use_ensemble=use_ensemble)

    if use_weak: 
        weak_model = fit_SINDy_model(train_data, alpha = sigma, threshold = thresh, dt = dt, t_data=t, 
                                     use_weak=True,  weak_kwargs = weak_kwargs, use_ensemble=use_ensemble
                                     )
        
        if use_ensemble:
            model.optimizer.coef_list = weak_model.optimizer.coef_list
        model.optimizer.coef_ = weak_model.optimizer.coef_
     
    if use_ensemble: 
        model.optimizer.coef_ = np.median(model.optimizer.coef_list, axis=0)

    return model


# code originally from paretoset repo:
# https://github.com/tommyod/paretoset/blob/master/paretoset/algorithms_numpy.py
def paretoset_efficient(costs, distinct=True):
    """An efficient vectorized algorhtm.

    This algorithm was given by Peter in this answer on Stack Overflow:
    - https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    costs = costs.copy()  # The algorithm mutates the `costs` variable, so we take a copy
    n_costs, n_objectives = costs.shape

    is_efficient = np.arange(n_costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        this_cost = costs[next_point_index]

        # Two points `a` and `b` are *incomparable* if neither dom(a, b) nor dom(b, a).
        # Points that are incomparable to `this_cost`, or dominate `this_cost`.
        # In 2D, these points are below or to the left of `this_cost`.
        current_efficient_points = np.any(costs < this_cost, axis=1)

        # If we're not looking for distinct, keep points equal to this cost
        if not distinct:
            no_smaller = np.logical_not(current_efficient_points)
            equal_to_this_cost = np.all(costs[no_smaller] == this_cost, axis=1)
            current_efficient_points[no_smaller] = np.logical_or(
                current_efficient_points[no_smaller], equal_to_this_cost
            )

        # Any point is incomparable to itself, so keep this point
        current_efficient_points[next_point_index] = True

        # Remove dominated points
        is_efficient = is_efficient[current_efficient_points]
        costs = costs[current_efficient_points]

        # Re-adjust the index
        next_point_index = np.sum(current_efficient_points[:next_point_index]) + 1

    # Create a boolean mask from indices and return it
    is_efficient_mask = np.zeros(n_costs, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask

def sim_wrapper(model, x0,t, n=1):
    '''safe method for integrating forward'''
    # try: 
    #     xs = model.simulate(x0, t)[-n]
    # except ValueError as e:
    #     warnings.warn(f'Encountered integration error {e}.')
        
    integrator_kws={"method": "RK45"}
    xs = model.simulate(x0, t, integrator_kws=integrator_kws)[-n]
    return xs


def predict_next_steps_sindy(model, x0s, dt, n=1):
    t = np.linspace(0,n*dt,n+1)
    return np.array([sim_wrapper(model, x0,t, n=n) for x0 in x0s])