import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from jax import random

from sparse_rev.data_gen import data_gen_scale
from sparse_rev.fit_utils import fit, pred_next_step
from sparse_rev.odes import lorenz


def get_weak_preds(model, x0, t):
    model.optimizer.coef_ = np.median(model.optimizer.coef_list, axis=0)
    integrator_kws={"method": "RK45"}
    med_preds = model.simulate(x0, t,  integrator_kws=integrator_kws)    
    return med_preds

def get_sindy_preds(model, x0, t):
    integrator_kws={"method": "RK45"}
    preds = model.simulate(x0, t,  integrator_kws=integrator_kws)  
    return preds

def get_dmd_preds(model, x0, t):
    preds = model.simulate(x0, len(t)+1)
    return preds

def get_sr_preds(model, x0, t):
    dim_size = x0.shape[-1]
    def dyn_fn(t,x):
        '''wrapper for model'''
        dx = model.predict(x.reshape(-1,dim_size))
        return dx
    
    preds = solve_ivp(dyn_fn, y0 = x0, t_eval = t, 
                      t_span=[0, t.max()]).y.T
    return preds

def get_nn_preds(model, x0, t):
    traj = [x0]
    for dt in t:
        x = traj[-1]
        pred =  x + model(x)
        traj.append(pred)
    return np.array(traj)

def get_gp_preds(model, x0, t):
    
    gp_mean_preds = [x0.reshape(1,-1)]
    n_points = len(t)
    for k in tqdm(range(n_points - 1)):
        x = gp_mean_preds[-1]
        gp_mean_preds.append(pred_next_step('gp', models['gp'], x,dt,n=1, train_data=train_data))
    gp_mean_preds = np.array(gp_mean_preds)[:,0]
    return gp_mean_preds


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    
    from sparse_rev import _parent_dir
    
    _SAVE_DIR = os.path.join(_parent_dir, 'data')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Palatino']

    train_seed = 42
    n_traj = 5
    T_max = 2.0
    dt = 0.02
    T_warmup = 1.0
    N_warmup = int(T_warmup // dt)
    USE_LEGEND = False

    sigma = 0.10
    signal_scales = np.array([8.16,  8.40, 26.94])

    t = np.arange(0, T_warmup + T_max, dt)

    # -----------------------------------------------
    # setup data
    # -----------------------------------------------
    data_kwargs = dict(dt=dt, 
                        T_max=T_max+T_warmup, 
                        d=3,
                        gen='uniform',
                        sigmas = sigma* signal_scales, 
                        dyn_fn='lorenz', 
                        dyn_params = (10, 2.66667, 28)
                        )

    key, data = data_gen_scale(seed=train_seed, n_traj=n_traj, 
                        gen_kwargs = dict(minval = -50, maxval = 50), 
                        **data_kwargs)

    # held out initial point near the attractor.
    x0 = np.array([2.5, 4, 12])
    t_test = np.arange(0,20,dt)
    test_data = solve_ivp(lorenz, y0=x0, 
                            t_span=[0,20], 
                            t_eval = t_test,
                            ).y.T.reshape(1,-1,3)

    sigmas = sigma* signal_scales
    noise = random.normal(key, shape =test_data.shape) * sigmas

    trajs_noise = test_data + noise

    test_data = {'clean': test_data,
                'noise': trajs_noise}

    t_idx = 0
    t_finish = 500
    x0 = test_data['clean'][0, t_idx:t_idx+1][0]
    t_eval = t_test[t_idx:t_finish] - t_test[t_idx]


    # -----------------------------------------------
    # fit models
    # -----------------------------------------------
    models = {}
    model_names = [
                    'sindy', 
                    'pysr', 
                'nn', 
                'dmd', 
                'gp', 
                'weak_sindy'
                ]
    for model_name in tqdm(model_names):
        train_data = data['noise'][:,N_warmup:]
        model = fit(model_name, train_data, sigma=sigma, t=t[N_warmup:], dt=dt, seed = 2000)
        models[model_name] = model
        
    # -----------------------------------------------
    # rollout models
    # -----------------------------------------------
    weak_preds = get_weak_preds(models['weak_sindy'], x0, t_eval)
    sindy_preds = get_sindy_preds(models['sindy'], x0, t_eval)
    nn_preds = get_nn_preds(models['nn'], x0, t_eval)
    sr_preds = get_sr_preds(models['pysr'], x0, t_eval)
    dmd_preds = get_dmd_preds(models['dmd'], x0, t_eval)
    gp_mean_preds = get_gp_preds(models['gp'], x0, t_eval)
    

    # -----------------------------------------------
    # configure plot
    # -----------------------------------------------
    fig, axes = plt.subplots(6,1, figsize=(5,5))
    pointsize=1
    linewidth=3
    colors = plt.cm.tab10.colors

    dynamics_idx = 0

    # plot the test data as background
    for i, ax in enumerate(axes):
        ax.plot(t_eval, test_data['clean'][0, t_idx:t_finish, dynamics_idx], c = 'gray', linewidth = 1, alpha = 0.5)
        ax.scatter(t_eval, test_data['noise'][0,t_idx:t_finish, dynamics_idx], c = 'gray', s = pointsize, alpha  = 0.25)
        ax.tick_params('both', labelsize=15)
        if i != 5:
            ax.set_xticks([])
        
        ax.set_yticks([])
        ax.set_xlim(-0.1,5.1)
        ax.set_ylim(-20,20)

    # plot all the predictions
    axes[0].plot(t_eval, dmd_preds[:-1,dynamics_idx], 
                c = colors[2], linewidth = linewidth)
    axes[1].plot(t_eval, nn_preds[:-1,dynamics_idx], 
                c = colors[3], linewidth=linewidth)
    axes[2].plot(t_eval, gp_mean_preds[:,dynamics_idx], 
                c = colors[4], linewidth = linewidth)

    axes[3].plot(t_eval, sr_preds[:,dynamics_idx], 
                c = colors[1], linewidth=linewidth)
    axes[4].plot(t_eval, sindy_preds[:,dynamics_idx], 
                c = colors[0], linewidth=linewidth)

    axes[5].plot(t_eval, weak_preds[:,dynamics_idx], 
                c = colors[5], linewidth=linewidth)
    
    if USE_LEGEND:
        plt.legend(bbox_to_anchor=(1.0, 1.0))
    
    plt.tick_params('both', labelsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(_SAVE_DIR, 'rollouts.png'))
    plt.close()