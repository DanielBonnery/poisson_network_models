def get_flowers():
	import pandas as pd
	'flowers_txt_url = 'http://www.ecologia.ib.usp.br/iwdb/data/plant_pollinator/text/mc_mullen.txt'
	'flowers_xls_url = 'http://www.ecologia.ib.usp.br/iwdb/data/plant_pollinator/excel/mcmullen_1993.xls'


	df1 = pd.read_excel(flowers_xls_url)
	df2 = pd.read_table(flowers_txt_url,  header=0)
	
	return df1,df2
