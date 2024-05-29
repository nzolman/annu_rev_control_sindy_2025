import jax.numpy as jnp
from jax import random

from scipy.integrate import solve_ivp

from sparse_rev import odes

NOISE_LEVELS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]
N_TRAJS = [2,4,8,16,32,64]

def data_gen(n_traj=16, dt=0.1, T_max=1,  d=2,
            gen = 'uniform',
            gen_kwargs = dict(x0_low = 1, x0_high = 4), 
            sigma = 1e-2, seed=0, dyn_fn = 'lotka', dyn_params = ([2./3.],)):
    '''
    Produce the Lotka-Volterra Training Set
    Arguments:
        `n_traj`: (int)
            Number of training trajectories
        `dt`: (float)
            timestep for evaluating solution
        `T_max`: (float)
            maximum time for each trajecory
            
        gen_kwargs: (dict)
            keyword arguments for generating function. e.g. if gen == 'diag':
                `x0_low`: (float)
                    initial condition lower boundary
                `x0_high`: (float)
                    initial condition upper boundary
        `sigma`: (float)
            fraction of max noise added (default 1e-2 -> 1% noise)
        `seed`: (int)
            Random Seed
    Returns:
        key: (jax.random.PRNG key)
        data: (dict)
            dictionary of clean and noisy data.
    '''
    key = random.PRNGKey(seed)

    T = jnp.arange(0,T_max,dt)

    if gen == 'diag': 
        x0_trains = jnp.array([[x0,x0] for x0 in jnp.linspace(gen_kwargs['x0_low'],
                                                              gen_kwargs['x0_high'], 
                                                              n_traj)])
    elif gen == 'uniform': 
        x0_trains = random.uniform(key, shape=(n_traj,d), 
                                    minval=gen_kwargs['minval'],
                                    maxval=gen_kwargs['maxval']
                                    )
        key,_ = random.split(key)
        
    dyn_fn = getattr(odes, dyn_fn)
    trajs = jnp.array([solve_ivp(dyn_fn, t_span = [T.min(), T.max()], y0=x0, t_eval = T, args=dyn_params
                                 ).y.T 
                       for x0 in x0_trains])

    noise = sigma * random.normal(key, shape =trajs.shape)

    trajs_noise = trajs + noise
    
    data = {'clean': trajs,
            'noise': trajs_noise}
    return key, data


# def data_2D(n_traj=5, dt=0.01, T_max=20, 
#             gen = 'diag',
#             gen_kwargs = dict(x0_low = 1, x0_high = 20), 
#             sigma = 1e-2, seed=0, dyn_fn = 'lotka', dyn_params = ()):
#     '''
#     Produce the Lotka-Volterra Training Set
#     Arguments:
#         `n_traj`: (int)
#             Number of training trajectories
#         `dt`: (float)
#             timestep for evaluating solution
#         `T_max`: (float)
#             maximum time for each trajecory
            
#         gen_kwargs: (dict)
#             keyword arguments for generating function. e.g. if gen == 'diag':
#                 `x0_low`: (float)
#                     initial condition lower boundary
#                 `x0_high`: (float)
#                     initial condition upper boundary
#         `sigma`: (float)
#             fraction of max noise added (default 1e-2 -> 1% noise)
#         `seed`: (int)
#             Random Seed
#     Returns:
#         key: (jax.random.PRNG key)
#         data: (dict)
#             dictionary of clean and noisy data.
#     '''
#     key = random.PRNGKey(seed)

#     T = jnp.arange(0,T_max,dt)

#     if gen == 'diag': 
#         x0_trains = jnp.array([[x0 for i in range(d)] for x0 in jnp.linspace(gen_kwargs['x0_low'],
#                                                               gen_kwargs['x0_high'], 
#                                                               n_traj)])
#     elif gen == 'uniform': 
#         x0_trains = random.uniform(key, shape=(n_traj,d), 
#                                     minval=gen_kwargs['minval'],
#                                     maxval=gen_kwargs['maxval']
#                                     )
#         key,_ = random.split(key)
        
#     dyn_fn = getattr(odes, dyn_fn)
#     trajs = jnp.array([solve_ivp(dyn_fn, t_span = [T.min(), T.max()], y0=x0, t_eval = T, args=dyn_params
#                                  ).y.T 
#                        for x0 in x0_trains])

#     noise = sigma * random.normal(key, shape =trajs.shape)

#     trajs_noise = trajs + noise
    
#     data = {'train_clean': trajs,
#             'train_noise': trajs_noise}
#     return key, data