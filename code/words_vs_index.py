# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:29:06 2019

@author: User
"""


# General packages for text reading
#import requests
import pandas as pd
import os


# Packages for text preprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk import bigrams
import itertools
import collections
import re

#import seaborn as sns
import pickle
import random
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nltk
import networkx as nx

def remove_sw(words,sw_list):
    return [word for word in words if word not in sw_list]

scriptPath = 'C:/Users/User/Desktop/texto_conpes'


# Absolute working directory
# Necessary for relative call inside the next loops
os.chdir(scriptPath)

# Connect to postgres
import psycopg2

try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "D1iv8Mv2a135nrW7kYB1",
                                  host = "nps-demo-instance.cmbnhcfpuqng.us-east-2.rds.amazonaws.com",
                                  port = "5432",
                                  database = "Team28_FP")

    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print ( connection.get_dsn_parameters(),"\n")
    sql_query = """
    SELECT
    c.dt_publicacion,
    tc.numero,
    c.titulo,
    tp.description,
    tc.texto,
    tc.textolimpio,
    p.presidente
    FROM
    CONPES C
    JOIN tipoconpes tp on c.idtipoconpes = tp.idtipoconpes
    JOIN PRESIDENTES P
    ON C.DT_PUBLICACION >= P.INICIO AND C.DT_PUBLICACION <= P.FIN
    JOIN textconpes tc ON tc.idtipoconpes = tp.idtipoconpes AND tc.numero = c.numero
    ORDER BY TC.NUMERO DESC
"""
    

    df_conpes = pd.read_sql(sql_query , connection)

    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    postgreSQL_select_Query = "select * from textconpes"
    #cursor.execute(postgreSQL_select_Query)
    cursor.execute(sql_query)
    record = cursor.fetchone()
    conpes_text_2 = cursor.fetchall() 
    print("You are connected to - ", record,"\n")

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


#------------------------------------------------------
#Complete database with Duques Conpes
try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "D1iv8Mv2a135nrW7kYB1",
                                  host = "nps-demo-instance.cmbnhcfpuqng.us-east-2.rds.amazonaws.com",
                                  port = "5432",
                                  database = "Team28_FP")

    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print ( connection.get_dsn_parameters(),"\n")
    
    #df_conpes = pd.read_sql(sql_query , connection)

    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    postgreSQL_select_Query = "select * from textconpes"
    cursor.execute(postgreSQL_select_Query)
    #cursor.execute(sql_query)
    record = cursor.fetchone()
    conpes_text_3 = cursor.fetchall() 
    print("You are connected to - ", record,"\n")

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
            
            

conpes_text = pd.DataFrame(conpes_text_2)


# Reset colnames
conpes_text.columns = ['fecha','numero','titulo','tipo','texto','textolimpio','presidente']

conpes_text.set_index('fecha', inplace=True)
conpes_text.index = pd.to_datetime(conpes_text.index)

test = conpes_text.resample('Y', kind='period', convention='start').agg({'texto': lambda x: ' '.join(x)})

# Datos abiertos

conteo_1 = []
for index, row in test.iterrows():
    temp = row['texto'].count('datos abiertos')
    conteo_1.append(temp)

sample = {'year':test.index, 'count':conteo_1} 

datos_abiertos = pd.DataFrame(sample)

no_she = [0]
no_she = no_she * 52
no_she[46] = 26.71
no_she[47] = 32.38
no_she[48] = 45.39
no_she[49] = 51.65
no_she[50] = 80
no_she

sample_2 = {'year':test.index, 'index':no_she}
datos_abiertos_2 = pd.DataFrame(sample_2)


datos_abiertos = datos_abiertos.set_index(datos_abiertos.year)
datos_abiertos_2 = datos_abiertos_2.set_index(datos_abiertos_2.year)

fig, ax = plt.subplots()
datos_abiertos.plot(ax=ax)
datos_abiertos_2.plot(ax=ax)
plt.show()
plt.title('Open Data Barometer vs ocurrences of "datos abiertos" in CONPES documents ')


#############################################3


conteo_2= []
for index, row in test.iterrows():
    temp = row['texto'].count('desigualdad')
    conteo_2.append(temp)
    
sample_3 = {'year':test.index, 'count':conteo_2}
desigualdad = pd.DataFrame(sample_3)
desigualdad = desigualdad.tail(9)

no_she_2 = [27.5 , 30.1, 26.6, 25.7, 24.1, 23.7, 22.6, 22.9, 22.4]

sample_4 = {'year':desigualdad.year, 'index':no_she_2}
desigualdad_2 = pd.DataFrame(sample_4)

desigualdad = desigualdad.set_index(desigualdad.year)
desigualdad_2 = desigualdad_2.set_index(desigualdad_2.year)

fig, ax = plt.subplots()
desigualdad.plot(ax=ax)
desigualdad_2.plot(ax=ax)
plt.show()
plt.title('Human Inequality Index vs ocurrences of "desigualdad" in CONPES documents ')



#############################################

conteo_3= []
for index, row in test.iterrows():
    temp = row['texto'].count('medio ambiente')
    conteo_3.append(temp)
    
sample_5 = {'year':test.index, 'count':conteo_3}
internet = pd.DataFrame(sample_5)
internet = internet.tail(9)
internet = internet.head(7)

no_she_3 = [1.3*100,1.4*100,1.4*100,		1.6*100,		1.7*100,		1.6*100,		1.8*100]

sample_6 = {'year':internet.year, 'index':no_she_3}
internet_2 = pd.DataFrame(sample_6)

internet = internet.set_index(internet.year)
internet_2 = internet_2.set_index(internet_2.year)

fig, ax = plt.subplots()
internet.plot(ax=ax)
internet_2.plot(ax=ax)
plt.show()
plt.title('Gender inequality vs ocurrences of "género" in CONPES documents ')

pd.DataFrame(conteo_3).plot()
