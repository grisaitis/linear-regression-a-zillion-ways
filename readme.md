# linear regression, implemented many ways

this demonstrates many ways of fitting a linear model. 

the point here is to showcase not linear regression per se, but rather different ways of fitting models in general. i just work with linear regression because it's easy, and i can get fancy with various (sometimes complicated) model fitting algorithms. 

by "model fitting algorithms" i mean:
- analytically calculating the least squares fit using linear algebra (i.e. ![too lazy for latex here](https://wikimedia.org/api/rest_v1/media/math/render/svg/46cf247a57b181c36165a0b6ae5ede6bdc1a24a3 "beta_hat")
- approximating the least squares fit with gradient descent (with ordinary least squares as a loss function, or negative log likelihood as well)
- approximating the predictive distribution using monte carlo sampling (and maybe variational calculus, if i have time) (aka bayesian regression)


### types of modeling
- regression
- classification
- others? rank, ordinal / level

### record trials
- name
- data used
- coefficient estimates (MLE, MAP, median posterior)
- objective function, and value
- time to execute
- framework name

### data
- fake data (given (N, p))
- titanic dataset?

### point estimation

#### optimize an objective function
- analytically
  - OLS for linear regression
- gradient descent
  - first order gradient
  - second order gradient (with exact hessian)
  - second order gradient (with approx hessian)
  - proximal gradient descent for lasso?


#### approximate the posterior
- monte carlo
  - basic
  - importance sampling
  - rejection sampling
  - (adaptive something?)
- markov chain monte carlo
  - metropolis-hastings
  - gibbs
  - hamiltonian
    - no u-turn

#### variational
- ???

### objective functions
- (regression) squared error
- (regression) negative log likelihood
- (regression) absolute error
- (classification) cross entropy
- (ranking / ordinal) ???

### models
- linear regression
- linear regression with L2 penalty (ridge) / gaussian prior
- linear regression with L1 penalty (lasso) / laplacian prior
- linear regression with L1 and L2 penalty (elastic net) / mixed prior
- logistic regression
- generalized linear model

### frameworks
- sklearn
- statsmodels
- pure python
- numpy + scipy
- jax
- numba?
- tensorflow
- pytorch
- pystan
- pymc3
- (R) rstanarm

### platforms
- cpu
- gpu

### math
- hessian is not always positive definite... when is it not for negative log likelihood?

### visualization
- for optimization:
  - the objective function surface in coefficient space
  - point evaluations by iteration
- for monte carlo:
  - the posterior as it samples
  - mode, median, mean points of posterior
