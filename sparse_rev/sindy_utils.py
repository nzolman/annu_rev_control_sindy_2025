import pysindy as ps
import numpy as np

from sklearn.model_selection import  LeaveOneOut, ShuffleSplit
from kneed import KneeLocator 

# def fit_SINDy_model(train_data, dt, T0, Tf, multiple_traj=True, opt=None, cubic =True, fd_order =3):
#     library_functions = [lambda x: x, lambda x,y: x * y, lambda x,y,z: x*y*z]
#     library_function_names = [lambda x: f'{x}', lambda x,y: f'{x}*{y}', lambda x,y, z: f'{x}*{y}*{z}']
    
#     if not cubic:
#         library_function_names.pop()
#         library_functions.pop()

#     # Try SINDy First
#     if multiple_traj:
#         u_dot_seq = []
#         for i in range(train_data.shape[0]):
#             u_dot_seq.append(ps.FiniteDifference(fd_order)._differentiate(train_data[i], t=dt))

#         train_data_seq = [train_data[i] for i in range(train_data.shape[0])]

#         ode_lib = ps.WeakPDELibrary(
#             library_functions=library_functions,
#             function_names=library_function_names,
#             interaction_only=False,
#             spatiotemporal_grid=np.arange(T0,Tf,dt),
#             include_bias=True,
#             is_uniform=True,
#             )
#         model = ps.SINDy(feature_library=ode_lib, optimizer = opt)
#         model.fit(train_data_seq, t=dt, multiple_trajectories = True)
#     else:
#         raise Exception('single traj not implemented')
#     return model


def cv_stlq_sindy(train_data, alpha = 1e-2, threshold = 1e-1, dt = 0.1, t=np.arange(0,20,0.1), n_splits=10, seed=0):
    '''assume train data has at least 2 trajs.'''
    alpha = max(alpha, 1e-6)
    
    optimizer = ps.STLSQ(alpha = alpha, threshold=threshold)
    lib = ps.PolynomialLibrary(degree=3)
    n_dim = train_data.shape[-1]

    # loo = LeaveOneOut()
    # splits = loo.split(train_data)
    n_splits = min(len(train_data), n_splits)
    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=.2, random_state=seed)
    splits = shuffle_split.split(train_data)
    
    mse_losses = []
    l0_losses = []
    for train_idx, val_idx in splits:
        # define model with given optimizer kwargs
        
        
        model = ps.SINDy(feature_library=lib, optimizer=optimizer, t_default=dt)
        model.fit(list(train_data[train_idx]), t=dt, multiple_trajectories=True, quiet=True)
        # predict on validations set
        val_pred = model.predict(train_data[val_idx])
        
        # figure out the "true" derivatives 
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
                 t = np.arange(0,1,10), threshold_list = np.logspace(-5,0,10)):
    sigma = max(sigma, 1e-6)
    
    # perform cross validation on each of the prospective thresholds
    res = np.array([cv_stlq_sindy(train_data, threshold=thresh, alpha = sigma, dt = dt, t=t)
                    for thresh in threshold_list])
    
    # find the pareto-optimal solution between accuracy and sparsity
    mses, l0s = res.mean(axis=1).T
    kneedle = KneeLocator( l0s,mses, S=1.0, curve="convex", direction='decreasing', online=True )

    
    if kneedle.elbow is None:
        elbow_idxes = np.array([[np.argmin(mses)]])
    else:
        elbow_idxes = np.where(l0s == kneedle.elbow)[0] 
    best_mse_idx = np.argmin(mses[elbow_idxes])

    best_elbow_idx = elbow_idxes[best_mse_idx]
    
    thresh = threshold_list[best_elbow_idx]
    
    # fit the model with the new value
    optimizer = ps.STLSQ(alpha = sigma, threshold=thresh)
    lib = ps.PolynomialLibrary(degree=3)

    model = ps.SINDy(feature_library=lib, optimizer=optimizer ,t_default=dt)
    model.fit(list(train_data), t=dt, multiple_trajectories=True)
    
    return model

def predict_next_steps_sindy(model, x0s, dt, n=1):
    t = np.linspace(0,n*dt,n+1)
    return np.array([model.simulate(x0, t)[-1] for x0 in x0s])