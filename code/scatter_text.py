# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:19:26 2019

@author: User
"""

%matplotlib inline
import scattertext as st
import re, io
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
from scipy.stats import rankdata, hmean, norm
import spacy
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer
display(HTML("&lt;style>.container { width:98% !important; }&lt;/style>"))

import es_core_news_sm
import en_core_web_sm
from spacy.lang.es.examples import sentences 


nlp = es_core_news_sm.load()

scriptPath = 'C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo'


# Absolute working directory
# Necessary for relative call inside the next loops
os.chdir(scriptPath)


# Load saved object of titles
with open('C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo\\titles_clean', 'rb') as titles_2_clean_file:
 
    # Step 3
    titles_2_clean = pickle.load(titles_2_clean_file)
 
    # After config_dictionary is read from file
    print(titles_2_clean)
    
tabla_titulos = pd.read_csv('C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo\\tabla_titulos.csv')

tabla_titulos['tituloclean'] = titles_2_clean

# Define categories - Divide between politica
conpes_credito = tabla_titulos[tabla_titulos['titulo'].str.contains("favorable")]
conpes_credito['type'] = 'credito'

conpes_politica = tabla_titulos[tabla_titulos['titulo'].str.contains("política")]
conpes_politica['type'] = 'política'

frames = [conpes_credito, conpes_politica]
tabla_conpes = pd.concat(frames)

# Parse
tabla_conpes['parsed'] = tabla_conpes.tituloclean.apply(nlp)

tabla_titulos['idtipoconpes'] = tabla_titulos['idtipoconpes'].replace(2,'social')
tabla_titulos['idtipoconpes'] = tabla_titulos['idtipoconpes'].replace(1,'económico')

corpus = st.CorpusFromParsedDocuments(tabla_conpes, category_col='type', parsed_col='parsed').build()

html = produce_scattertext_explorer(corpus,
                                    category='credito',
                                    category_name='Crédito',
                                    not_category_name='Política',
                                    width_in_pixels=1000,
                                    minimum_term_frequency=1,
                                    transform=st.Scalers.scale,
                                    metadata=tabla_conpes['numero'])

file_name = 'Scattertext-conpestype.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)



# Escala logarítmica

html = produce_scattertext_explorer(corpus,
                                    category='credito',
                                    category_name='Crédito',
                                    not_category_name='Política',
                                    width_in_pixels=1000,
                                    minimum_term_frequency=1,
                                    transform=st.Scalers.log_scale_standardize)
                                    #metadata=tabla_titulos['numero'])


file_name = 'Scattertext-conpestype-log.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)


# Escala por percentiles


html = produce_scattertext_explorer(corpus,
                                    category='credito',
                                    category_name='Crédito',
                                    not_category_name='Política',
                                    width_in_pixels=1000,
                                    minimum_term_frequency=1,
                                    transform=st.Scalers.percentile,
                                    metadata=tabla_conpes['numero'])

file_name = 'Scattertext-conpestype-percentile.html'
open(file_name, 'wb').write(html.encode('utf-8'))


conpes_declaración = tabla_titulos[tabla_titulos['titulo'].str.contains("estratégica")]
