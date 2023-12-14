<img src="docs/source/mypy_light.svg" alt="mypy logo" width="300px"/>

Project for the Sequential Monte Carlo course: Accelerating Bayesian estimation for network Poisson models using frequentist variational estimates
=======================================

[![Stable Version](https://github.com/DanielBonnery/?color=blue)](https://github.com/DanielBonnery/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Get data
-------------


```python
def get_flowers():
	import pandas as pd
	'flowers_txt_url = 'http://www.ecologia.ib.usp.br/iwdb/data/plant_pollinator/text/mc_mullen.txt'
	'flowers_xls_url = 'http://www.ecologia.ib.usp.br/iwdb/data/plant_pollinator/excel/mcmullen_1993.xls'


	df1 = pd.read_excel(flowers_xls_url)
	df2 = pd.read_table(flowers_txt_url,  header=0)
	
	return df1,df2
```


