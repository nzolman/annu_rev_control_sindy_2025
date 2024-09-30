import numpy as np
import sympy
from pysr import PySRRegressor
from pysindy.differentiation import FiniteDifference
from sklearn.model_selection import LeaveOneOut
from kneed import KneeLocator 
from scipy.integrate import solve_ivp

from sparse_rev import _tmp_dir

def simple_pysr_fit(train_data, parsimony = 0.0032, t=np.arange(0,20,0.1), 
                    n_iter = 10, verbosity=0, tmp_dir = _tmp_dir):
    '''
    Fit PySR model
    
    Arguments:
        - train_data (ndarray)
            training data
        - parsimony (float)
            pysr parsimony parameter, encourages simplicity in equations
        - t (ndarray)
            time mesh (used for approximating dx/dt)
        - n_iter (int)
            number of iterations to use
        - verbosity (int)
            pysr verbosity level
        - tmp_dir (str)
            Temporary directory
    '''
    fd_method = FiniteDifference(axis=-2)
    
    X_dot = fd_method(train_data, t)
    model = PySRRegressor(
        niterations=n_iter,
        binary_operators=["+", "*"], # sticking to polynomials
        unary_operators=[
        ],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        tempdir=tmp_dir,
        temp_equation_file=True,
        parsimony=parsimony, 
        warm_start=False,
        verbosity=verbosity,
        random_state=42,    # https://github.com/MilesCranmer/PySR/discussions/386#discussioncomment-6493277
        deterministic=True,
        procs=0,
        multithreading=False,
    )

    X = train_data
    model.fit(np.concatenate(X), 
              np.concatenate(X_dot))
    return model

def cv_pysr_parsimony(train_data, parsimony = 0.0032, t=np.arange(0,20,0.1), n_iter = 10, verbosity=0,
                      tmp_dir = _tmp_dir):
    
    '''
    Cross-validation to select the parsimony parameter.
    
    Note: assume train data has at least 2 trajs. We'll do leave one out cv
    
    Arguments:
        - train_data (ndarray)
            training data
        - parsimony (float)
            pysr parsimony parameter, encourages simplicity in equations
        - t (ndarray)
            time mesh (used for approximating dx/dt)
        - n_iter (int)
            number of iterations to use
        - tmp_dir (str)
            Temporary directory
    '''
    fd_method = FiniteDifference(axis=-2)
    
    X_dot = fd_method(train_data, t)
    n_dim = train_data.shape[-1]
    
    loo = LeaveOneOut()
    splits = loo.split(train_data)
    
    mse_losses = []
    complexities = []
    
    model = PySRRegressor(
        niterations=n_iter,
        binary_operators=["+", "*"], # sticking to polynomials
        unary_operators=[
        ],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        tempdir=tmp_dir,
        temp_equation_file=True,
        parsimony=parsimony, 
        warm_start=False,
        verbosity=verbosity
    )
    
    for train_idx, val_idx in splits:
        # define model with given optimizer kwargs
        
        x = train_data[train_idx]
        x_dot = X_dot[train_idx]
        model.fit(np.concatenate(x), 
                  np.concatenate(x_dot))
        
        # predict on validations set
        val_pred = model.predict(np.concatenate(train_data[val_idx]))
        
        # figure out the "true" derivatives 
        xdot_val = X_dot[val_idx]
        
        # reshape data
        val_pred = val_pred.reshape(-1, n_dim)
        xdot_val = np.array(xdot_val).reshape(-1,n_dim)
        
        # calculate losses
        mse_loss = float(np.mean((val_pred - xdot_val)**2))
        
        mse_losses.append(mse_loss)
        complexities.append(model.get_best()[0].complexity)
        
    return np.array([mse_losses, complexities]).T

def pareto_pysr(train_data, dt =0.1, t = np.arange(0,1,10), p_space = np.logspace(-5,1,10), 
                tmp_dir = _tmp_dir):
    '''
    Do a pareto sweep using cross validation and refit the model
    '''
    # use cross-validation at each of the prospective parsimony parameters
    res = np.array([cv_pysr_parsimony(train_data, parsimony = p, t=t, n_iter = 10, verbosity=0) for p in p_space])
    
    # calculate the pareto-optimal parsimony value
    mses, complexities = res.mean(axis=1).T
    kneedle = KneeLocator(complexities, mses, S=1.0, curve="convex", direction='decreasing', online=True )
    parsimony = p_space[np.where(complexities == kneedle.elbow)[0][0]]
    
    model = simple_pysr_fit(train_data, parsimony = parsimony, t=t, n_iter = 10, verbosity=0,
                    tmp_dir = _tmp_dir)
    
    return model

def predict_next_steps_pysr(model, x0s, dt, n=1):
    '''
    Predict the next step for each x0 in a list of x0s.
    '''

    def dyn_fn(t,x):
        '''wrapper for model'''
        dx = model.predict(x.reshape(-1,x.shape[-1]))
        return dx
    
    t = np.linspace(0,n*dt,n+1)
    sols = [solve_ivp(dyn_fn, y0 = x0, t_eval = t, t_span=[0, t.max()]
                      ).y.T[-n] 
            for x0 in x0s
            ]
    
    return np.array(sols)

def print_sr(model):
    return sympy.simplify(sympy.Matrix(model.sympy()))