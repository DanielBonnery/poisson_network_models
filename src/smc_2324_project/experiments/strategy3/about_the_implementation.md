# Implementation details

One will use `particles` to implement the part 2.3 of the paper (SMC sampler).

## Basics

The model is implemented using `TemperingBridge`. 
To define a `TemperingBridge` model, one needs to specify:
- the `.logtarget` method
- the `base_dist` attribute which must be a distribution

The SMC algorithm is handled by `AdaptiveTempering`.
`move` is defined using `MCMCSequence`.

## About `base_dist`

Depending on the strategy (from VEM or from prior), `base_dist` may change.
One needs to define `Dirichlet` (which does not exist) to define the prior on $\theta$ (using also `IndepProd` and a multivariate normal).
Then use an `IndepProd` of `Cond` to get the prior on $(Z,\theta)$.

**Warning: Z should be categorical**

## About `move`

During `.calibrate`, access the current exponent through the shared attributes of the model to update the MCMC kernel.

## Other details

By default, the criteria to choose the next rho does not use cESS, and there is a resampling at each step. Those details could be overriden later on.