import pysindy as ps
import numpy as np

def fit_dmd(train_data, sigma = 1e-2, dt = 0.1, t=np.arange(0,20,0.1)):
    '''
    Simple function for vanilla DMD fitting. Discrete-time SINDy reduces to 
    DMD when there is zero threshold and a linear library.
    
    Arguments:
    
    - train_data: (ndarray)
        training data
    - sigma: (float)
        L2 coefficient
    - dt: (float)
        time delta between steps
    - t: (ndarray)
        array of time values (placeholder)
    '''
    sigma = max(sigma, 1e-6)
    
    # zero threshold reduces down to normal L2-Regularized regression
    optimizer = ps.STLSQ(alpha = sigma, threshold=0)
    lib = ps.PolynomialLibrary(degree=1)
    n_dim = train_data.shape[-1]

    train = list(train_data)

    # define model with given optimizer kwargs
    model = ps.SINDy(feature_library=lib, optimizer=optimizer, t_default=dt, discrete_time=True)
    model.fit(train, t=dt, multiple_trajectories=True, quiet=True)
    
    return model