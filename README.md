# Description
Accompanying code for ["Machine Learning for Sparse Nonlinear Modeling and Control" by Brunton et al.](https://www.annualreviews.org/content/journals/10.1146/annurev-control-030123-015238) in Annual Review of Control, Robotics, and Autonomous Systems. 

**NOTE**: The intent for this code was to create a heuristic baseline comparison between dynamics regression methods. For each method, a choice was made for the implementation to try and provide a high-level illustration of the approaches. We recognize that there are modern variations of these methods that can provide substantial improvements; we encourage the community to try out their own methods on a wide variety of problems. 

There is no plan to continue to maintain this code after publication; however, please feel free to start a GitHub Discussion or Issue for any clarifying questions or comments.

# Installation:
To install the package, you must clone the repository

```
git clone https://github.com/nzolman/annu_rev_control_sindy_2025.git
```

and then run the following in the root directory: 

```bash
pip install -r requirements.txt
pip install -e .
```

# Running the Benchmarks
All scripts for producing Figure 4 in the paper can be found under `sparse_rev/scripts`. The sweeps are fairly time intensive and written to be single-threaded, so they may take a few hours to run to completion. However, they should be trivially parallelized by sweeping combinations of `(model_name, seed)` across independent workers. 

An example of the output from these sweeps can be found under the `data/` directory in CSV format.

**NOTE**: During your first usage, you might notice julia being downloaded due to using the pysr package.

# Methods

## SINDy
For SINDy, we use the [PySINDy package (v1.7.5)](https://github.com/dynamicslab/pysindy/tree/v1.7.5) [1]. We use a STRidge optimizer, which incorporates an $L^2$ penalty with sequentially thresholded least squares (STLS). Empirically, the optimization problem is insensitive to a wide-range of values for the $L^2$ penalty, so we fix the coefficient to be the scale of the noise added. For choosing the coefficient threshold, we perform cross validation (where the validation split uses independent _trajectories_) and use the [kneed](https://github.com/arvkevi/kneed) package [3] to select the pareto optimal fit that balances the cross-validation loss and the sparsity (i.e. $L^0$ loss). 

We use a continuous time model with a cubic library for estimating the dynamics, and use scipy's RK45 integrator to perform the next-state predictions.

## DMD
For DMD, we again use the PySINDy package. Discrete-time SINDy reduces to vanilla DMD when we restrict linear library and set the coefficient threshold to 0 in STRidge. We again fix the coefficient on the $L^2$ penalty to be the scale of the added noise.

## Weak Ensemble SINDy
For Weak Ensemble SINDy, we once again use the PySINDy package. We consider a continuous time model with a cubic library for estimating the dynamics and use scipy's RK45 integrator to perform the next-state predictions. For the Weak-SINDy implementation, we define the number of test functions and their width to depend on the length of the trajectory. For E-SINDy, we use 20 models with library ensembling and bagging, and utilize the median coefficients during inference. Just like with SINDy, we use cross-validation to select the threshold parameter. The $L^2$ coefficient is fixed to be $10^{-3}$ because the features are built out of inner products between the test functions and basis functions and the noise scale does not translate the same.

## Neural Networks
We utilize the [Equinox library](https://github.com/patrick-kidger/equinox) [4] for building our neural networks on top of [JAX](https://github.com/jax-ml/jax) [5]. To simplify the problem, we fit the neural net (NN) based off the regression $x_{n+1} = x_n + NN(x_n)$, i.e. we purely learn the discrete time update. Because of the simplicity of the problem, we restrict to a basic MLP network with 2 hidden layers of size 16 each and tanh activations. We use a learning rate of $10^{-4}$ and use a validation set to choose the best model during 10,000 epochs of training to avoid overfitting. 

## Gaussian Processes
We utilize the [GP JAX library](https://github.com/JaxGaussianProcesses/GPJax) [6] for fitting our Gaussian processes. We use a multi-dim polynomial kernel. Because we do end up using a large number of points, we utilize Sparse GPs and optimize the kernel hyperparameters using a stochastic optimization (following the example in the [GPJax docs](https://docs.jaxgaussianprocesses.com/_examples/collapsed_vi/)) and bootstrap inducing points through a validation set trajectory. Even for the low-data limit, we did not see a substantial drawback to utilizing Sparse GPs instead of regular GPs, so we kept it the same for all.

## Symbolic Regression
We utilize the [PySR library](https://github.com/MilesCranmer/PySR) [7] to perform our symbolic regression. We compose functions through addition and multiplication of terms (forming polynomials) and allow up to 10 solver iterations. Increasing the number of iterations might substantially improve the convergence of the coefficients and lead to better predictions, but it also greatly increases the amount of time to execute the code.

# References

[1] Kaptanoglu, Alan A., et al. "PySINDy: A comprehensive Python package for robust sparse system identification."

[2] Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations." Science advances 3.4 (2017): e1602614.

[3] Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.

[4] Kidger, Patrick, and Cristian Garcia. "Equinox: neural networks in JAX via callable PyTrees and filtered transformations." arXiv preprint arXiv:2111.00254 (2021).


[5] Bradbury, James, et al. "JAX: composable transformations of Python+ NumPy programs." (2018).

[6] Pinder, Thomas, and Daniel Dodd. "Gpjax: A gaussian process framework in jax." Journal of Open Source Software 7.75 (2022): 4455.

[7] Cranmer, Miles. "Interpretable machine learning for science with PySR and SymbolicRegression. jl." arXiv preprint arXiv:2305.01582 (2023).
