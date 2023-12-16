#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:06:36 2023

@author: daniel
"""
def get_fungal_species_dataset():
	import urllib.request
    import zipfile
	fungal_zip_url = 'https://projecteuclid.org/journals/supplementalcontent/10.1214/10-AOAS361/supzip_1.zip'	
    filehandle, _ = urllib.request.urlretrieve(fungal_zip_url )
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)
	
	return file
