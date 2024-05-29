import numpy as np

import jax
from jax import jit, random
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from optax.contrib import reduce_on_plateau

import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

def get_all_params(layers):
    weights = jnp.concatenate([layer.weight.flatten() for layer in layers])
    biases = jnp.concatenate([layer.bias.flatten() for layer in layers])
    
    return jnp.concatenate([weights, biases])

@jit
def l2_reg(layers):
    params = get_all_params(layers)
    return jnp.sum(params**2)

@eqx.filter_jit
def mse_loss(model,x,y):
    pred = x + jax.vmap(model)(x)
    mse = jnp.mean((pred - y)**2)
    return mse

@eqx.filter_jit
def loss(model, x, y, alpha = 1e-4):
    mse = mse_loss(model, x, y)
    reg = l2_reg(model.layers)
    
    return mse + alpha * reg

def epoch_idxes(n_tot, n_batches):
    rand_idxes = np.random.permutation(np.arange(n_tot))
    batch_idxes = jnp.split(rand_idxes, n_batches)
    return batch_idxes

def prep_data(train_data, val_split = 0.25):
    N_traj = len(train_data)
    n_traj_val = max(1, int(val_split* N_traj))

    X_train = train_data[:-n_traj_val, :-1]
    Y_train = train_data[:-n_traj_val, 1:]

    X_val = train_data[-n_traj_val:,:-1]
    Y_val = train_data[-n_traj_val:, 1:]

    X_train_flat = jnp.concatenate(X_train)
    Y_train_flat = jnp.concatenate(Y_train)

    X_val_flat = jnp.concatenate(X_val)
    Y_val_flat = jnp.concatenate(Y_val)
    return X_train_flat, Y_train_flat, X_val_flat, Y_val_flat

def train_nn(train_data, n_epochs=5000, n_batch=10,
             seed=0, lr=1e-4, l2_reg=1e-4, 
             use_plateau = False,
             plateau_kwargs = dict(patience=20, cooldown=10, 
                                   factor=0.5,rtol=1e-4, 
                                #    accumulation_size=200
                                   )
             ):
    X_train_flat, Y_train_flat, X_val_flat, Y_val_flat = prep_data(train_data)
    n_dim = X_train_flat.shape[-1]
    
    np.random.seed(seed) # for epoch idxes, numpy is faster than jax on cpu
    key = random.PRNGKey(seed)
    
    # make NN model
    model = eqx.nn.MLP(key = key, in_size=n_dim, out_size=n_dim, 
                   width_size=16, depth=2, activation=jnp.tanh)

    # make optimizer
    if use_plateau:
        optim = optax.chain(optax.adam(lr),
                            reduce_on_plateau(**plateau_kwargs)
                            )
    else:
        optim = optax.adam(lr)
    # init
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    

        
    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP,
        opt_state: PyTree,
        x: Float[Array, "batch n"],
        y: Int[Array, " batch n"],
        alpha = 1e-4,
    ):
        '''Single Gradient step'''
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y, alpha)
        updates, opt_state = optim.update(grads, opt_state, model, loss = loss_value)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit # this may not be possible to jit
    def train_epoch(model, opt_state, X_train, Y_train, splits, alpha=1e-4):
        '''Train through epoch'''
        # n_tot = len(X_train)
        # splits = epoch_idxes
        tot_loss = 0.0
        for idx in splits:
            x = X_train[idx]
            y = Y_train[idx]
            model, opt_state, loss_value = make_step(model, opt_state, x, y, alpha)
            tot_loss += loss_value
        return model, opt_state, tot_loss
    
    losses = []
    val_losses = []
    # TO-DO: Early stopping of validation loss
    for epoch in range(n_epochs):
        # get batch indices
        splits = epoch_idxes(len(X_train_flat), n_batch)
        
        # train for an epoch
        model, opt_state, tot_loss = train_epoch(model, opt_state,
                                                 X_train_flat, Y_train_flat, 
                                                 splits, alpha=l2_reg
                                                 )
        # compute validation loss
        val_loss = mse_loss(model, X_val_flat, Y_val_flat)
        val_losses.append(val_loss)
        losses.append(tot_loss)
    
    return model, losses, val_losses