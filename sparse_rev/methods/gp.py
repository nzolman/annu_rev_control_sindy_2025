# A significant portion of this code was taken from the gpjax example of modeling 2D vectorfields
# [1] https://docs.jaxgaussianprocesses.com/examples/oceanmodelling/
# This implementation seems pretty inefficient; probably a better way to do this with tensor products.
from dataclasses import dataclass, field

from jax import hessian, jit, config, random
import jax.numpy as jnp
from jaxtyping import Array, Float

import gpjax as gpx
import optax
import pysindy as ps


# Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
def label_position(data, n_dim=3):
    # introduce alternating z label
    n_points = len(data[0])
    label = jnp.tile(jnp.arange(n_dim), n_points)
    return jnp.vstack((jnp.repeat(data, repeats=n_dim, axis=1), label)).T

# change vectors y -> Y by reshaping the velocity measurements
def stack_velocity(data):
    return data.T.flatten().reshape(-1, 1)


def dataset_3d(pos, vel, n_dim=3):
    return gpx.Dataset(label_position(pos, n_dim=n_dim), stack_velocity(vel))

@dataclass
class Kernel2D(gpx.kernels.AbstractKernel):
    '''Adapted from [1] for testing'''
    
    # multiple kernels
    kernel0: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])
    )
    kernel1: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])
    )


    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0
        
        kernels = jnp.array([self.kernel0(X, Xp), self.kernel1(X, Xp)])
        
        # we use the last input to identify the label
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        
        
        # check if these are the same index, 
        # if they are, then we just take that kernel. 
        # if they aren't, we return zero.
        k0_switch = jnp.logical_and(z==zp, z==0)
        k1_switch = jnp.logical_and(z==zp, z==1)
        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)


@dataclass
class Kernel3D(gpx.kernels.AbstractKernel):
    '''
    For Lorenz, we use 3 kernels instead of 2. Adapted from [1]
    '''
    kernel0: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.Polynomial(degree=2, active_dims=[0, 1, 2])
    )
    kernel1: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.Polynomial(degree=2, active_dims=[0, 1, 2])
    )
    kernel2: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.Polynomial(degree=2, active_dims=[0, 1, 2])
    )

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # standard polynomial kernel is x and x' are on the same output, otherwise returns 0
        
        # we use the last input to identify the label
        z = jnp.array(X[3], dtype=int)
        zp = jnp.array(Xp[3], dtype=int)
        
        # check if these are the same index, 
        # if they are, then we just take that kernel. 
        # if they aren't, we return zero.
        k0_switch = jnp.logical_and(z==zp, z==0)
        k1_switch = jnp.logical_and(z==zp, z==1)
        k2_switch = jnp.logical_and(z==zp, z==2)

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp) + k2_switch * self.kernel2(X, Xp)


def initialise_gp(kernel, mean, dataset, obs_std=1e-3):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array([obs_std], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior


def optimise_mll(posterior, dataset, NIters=1000, key=None):
    # define the MLL using dataset_train
    objective = jit(gpx.objectives.ConjugateMLL(negative=True))
    # Optimise to minimise the MLL
    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=objective,
        train_data=dataset,
        verbose=False
    )
    return opt_posterior

def prep_data(train_data, use_diff = False):
    if not use_diff: 
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:]
    else:
        # regress on y(x) = Delta x
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:] - X_train
    
    X_train_flat = jnp.concatenate(X_train)
    Y_train_flat = jnp.concatenate(Y_train)
    
    # label and place the training data into a Dataset object to be used by GPJax
    dataset_train = dataset_3d(X_train_flat.T, Y_train_flat.T)
    return dataset_train



def fit_ker(all_data, sigma = 1e-3, key=None, use_sparse_gp=False, 
            n_inducing=5, z_bounds = [-50.0,50.0], n_iters = 100, seed=42, 
            use_diff = False, n_traj_val = 0, max_val = 100,
            sched_kwargs = dict(init_value=0.1,
                                peak_value=0.05,
                                warmup_steps=10,
                                decay_steps=50,
                                end_value=0.001,
                                )
            ):
    key = random.PRNGKey(seed)
    d = all_data.shape[-1]
    
    val_data = all_data[:n_traj_val]
    train_data = all_data[n_traj_val:]

    dataset_train = prep_data(train_data, use_diff=use_diff)
    
    # define kernel
    mean = gpx.mean_functions.Zero()
    kernel = Kernel3D()
    if not use_sparse_gp:
        velocity_posterior = initialise_gp(kernel, mean, dataset_train, obs_std=sigma)
        opt_velocity_posterior = optimise_mll(velocity_posterior, dataset_train, key=key)
        history = None
    else:
        # If there's too many points, it's beneficial to cast to a stochastic optimization
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset_train.n)
        prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
        posterior = prior * likelihood
        
        # If we're evaluating on the Lorenz attractor, might be better
        # to choose these induced points on/near the manifold as opposed to the ambient space
        if n_traj_val !=0:
            val_flat = jnp.concatenate(val_data)
            n_val = len(val_flat)
            val_idx = random.permutation(key, jnp.arange(n_val),independent=True)[:max_val]
            val_flat = val_flat[val_idx]
            key, _ = random.split(key)
            z_flat = label_position(val_flat.T, n_dim= d)
        else:
            z_lin = jnp.linspace(*z_bounds, n_inducing)
            z_mesh = jnp.meshgrid(*[z_lin for i in range(d)], jnp.arange(d))
            z_flat = jnp.array([z.flatten() for z in z_mesh]).T

        q = gpx.variational_families.CollapsedVariationalGaussian(posterior=posterior, 
                                                                inducing_inputs=z_flat)
        elbo = gpx.objectives.CollapsedELBO(negative=True)
        elbo = jit(elbo)

        schedule = optax.warmup_cosine_decay_schedule(
            **sched_kwargs
        )

        opt_velocity_posterior, history = gpx.fit(
            model=q,
            objective=elbo,
            train_data=dataset_train,
            optim=optax.adam(learning_rate=schedule),
            num_iters=n_iters,
            key=key,
            batch_size=128,
            verbose=False
        )
    return opt_velocity_posterior, history


def get_uncertain_trajs(model, x0, train_data, traj_len = 100):
    dataset_train = prep_data(train_data)
        
    @jit
    def sample(key, x):
        '''Sample the GP at a given point'''
        distr = model.predict(
                            label_position(x.reshape(-1,1)),
                            dataset_train
                                    )
        x_new = distr.sample(seed=key, sample_shape = (1,))[0]
        return x_new

    @jit
    def get_traj(key, x0):
        '''Rollout predictions by sampling the GP'''
        preds = [x0]
        for i in range(1, traj_len):
            preds.append(sample(key, preds[-1]))
            key, _ = random.split(key,)
        return key, jnp.array(preds)
    
    trajs = []
    
    for i in range(traj_len):
        key, x = get_traj(key, x0)
        trajs.append(x)
    trajs = jnp.array(trajs)
    return trajs