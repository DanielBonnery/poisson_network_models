<!-- <img src="docs/source/mypy_light.svg" alt="mypy logo" width="300px"/> -->

# Reimplementing *Accelerating Bayesian estimation for network Poisson models*

This project reimplements the paper https://hal.inrae.fr/hal-03202058 as a final project for the course "Hidden Markov models and sequential Monte-Carlo" taught by Nicolas Chopin.
The reimplementation uses `Python` and the package `particles`.

## Reproduction of the plots

```{sh}
git clone https://github.com/DanielBonnery/poisson_network_models" 
jupyter nbconvert --to notebook --execute "src/smc_2324_project/experiments/results_summary/plot_results.ipynb"
```

## Reproduction of all the code

