# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:35:19 2019

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

conpes_text = pd.DataFrame(conpes_text_2)


# Reset colnames
conpes_text.columns = ['fecha','numero','titulo','tipo','texto','textolimpio','presidente']

conpes_text.presidente.unique()
Santos_II =  conpes_text[conpes_text['presidente']=='Alvaro Uribe Velez I']
titles_2_clustering = Santos_II.titulo
titles_2_clustering = [' '.join(w for w in p.split() if w not in all_stopwords) for p in titles_2_clustering] # For clustering
titles_2_clustering = pd.DataFrame([' '.join(w for w in p.split() if w not in all_stopwords) for p in titles_2_clustering] ) # For knowledgegraph
titles_2_clustering.columns = ['titulo']


titles_2_clustering["titulo"]= titles_2_clustering["titulo"].replace('paz', "Paz") 
titles_2_clustering["titulo"]= titles_2_clustering["titulo"].replace('posconflicto', "Posconflicto") 

titles_2_clustering["titulo"]= titles_2_clustering["titulo"].replace('60', "") 
titles_2_clustering["titulo"]= titles_2_clustering["titulo"].replace('$', "") 

# SAVE THIS TEXT #
# file name
with open('C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo\\all_text_clean', 'wb') as all_text_clean:
 
  # save
  pickle.dump(titles_2_clustering_f, all_text_clean)
  

with open('C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo\\all_text_clean', 'rb') as clean_text_2:
 
    # Step 3
    titles_2_clustering_f = pickle.load(clean_text_2)


#corpus = clean_text.Text.copy()
corpus = titles_2_clustering.copy()
#corpus = titles_2_clean.copy()

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus)

#print(matrix)

a = matrix.todense()
print(a)

#--------------------------------------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(a)
dist_mat = np.asmatrix(dist)
dist_mat


#--------------------------------------------------------------------------------

# Attempt
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca_data = pca.fit_transform(dist_mat)

pca_all = pca.fit(dist_mat)
pca_all.explained_variance_ratio_
pca_all.explained_variance_ratio_.sum()

plt.figure(figsize=(9,8))
plt.scatter(pca_data[:,1], pca_data[:,2], c='goldenrod',alpha=0.5)
plt.show

#-------------------------------------------------------------------------------
# ELBOW METHOD FOR NUMBER OF CLUSTERS

# Save components to a DataFrame
PCA_components = pd.DataFrame(pca_data)
PCA_components.iloc[:,9]

ks = range(1, 20)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_jobs = -1, n_clusters=k, init='k-means++')
    
    # Fit model to samples
    model.fit(PCA_components)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# SILHOUETTE

from sklearn.metrics import silhouette_score


random.seed(100)
sil = []
kmax = 20

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
#%matplotlib inline
#%matplotlib auto

for k in range(2, kmax+1):
  kmeans = KMeans(n_jobs = -1 ,n_clusters = k, init='k-means++',random_state=1).fit(PCA_components)
  labels = kmeans.labels_
  sil.append(silhouette_score(PCA_components, labels, metric = 'euclidean'))


plt.plot(ks,sil,  '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('silhouette score')
plt.xticks(ks)
plt.show()

# With kmeans ++ the silhouette returns

#### Optimal number with WCSS
from math import sqrt

def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    
    return wcss

wcss = calculate_wcss(PCA_components)

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

optimal_number = optimal_number_of_clusters(wcss)

#--------------------------------------------------------------------------------

#Set a 3 KMeans clustering
kmeans = KMeans(n_clusters = 6)

#Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(PCA_components)

#Define our own color map
LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b', 3: 'y', 4: 'peru', 5:'coral', 6:'olive',7:'cyan',8:'black'}
LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b', 3: 'y', 4: 'peru', 5:'coral'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(PCA_components.iloc[:,1],PCA_components.iloc[:,2], c= label_color, alpha=0.5) 
plt.show()

#-----------------------------------------------------------------------------------

from sklearn.manifold import TSNE
X = TSNE(n_components=2, perplexity = 10).fit_transform( pca_data )

#plot graph
#colors = np.array(['red','green','blue','yellow','peru','coral','olive','cyan','black'])
colors = np.array(['red','green','blue','yellow','peru','coral'])
plt.scatter(X[:,0], X[:,1], c=colors[kmeans.labels_])
plt.title('K-Means (t-SNE) -  Uribe I')
plt.show()    


# Extract documents from clusters

documents_clustered = pd.DataFrame(titles_2_clustering, X_clustered, columns = ['titulo'])
documents_clustered['cluster'] = documents_clustered.index
#documents_clustered = documents_clustered.reset_index()

documents_clustered = documents_clustered.sort_values(by=['cluster'])

documents_grouped = documents_clustered.groupby('cluster').agg(' '.join)
documents_grouped['cluster'] = np.arange(len(documents_grouped))


# Most common words per cluster
most_words = pd.DataFrame({"KeyWords" : documents_grouped["titles"].apply(lambda x: [k for k, v in Counter(x).most_common(50)])}) 

# TOTAL WORDS
Counter(" ".join(documents_grouped["titles"]).split()).most_common(50)


words_per_cluster = []

for i in range(len(documents_grouped)):
    temp = Counter(documents_grouped['titulo'][i].split()).most_common(20)
    words_per_cluster.append(temp)
    
# Knowledge graphs

tokens = []

for i in range(len(documents_grouped)):
    temp = nltk.tokenize.word_tokenize(documents_grouped['titulo'][i])
    tokens.append(temp)



# WORD NETWORK OF BIGRAMS
    
def bigram_network(cluster):
    ngram = list(ngrams(cluster, 2))
    ngram_frequency = nltk.FreqDist(ngram)
#ngram_frequency.plot(30,cumulative=False)

    ngram_frequency = dict(ngram_frequency.copy())
    ngram_table = pd.DataFrame.from_dict(ngram_frequency, orient='Index', columns = ['frequency'])
    ngram_table.reset_index(level=0, inplace=True)
    # Order the table
    ngram_table = ngram_table.sort_values(by=['frequency'], ascending = False)

    terms_bigram = [list(bigrams(cluster))]
    bigrams_l = list(ngrams(cluster, 2))
    bigrams_l = list(itertools.chain(*terms_bigram))
    
    bigram_counts = collections.Counter(bigrams_l)
    bigram_counts_graph = bigram_counts.most_common(30)
    bigram_df = pd.DataFrame(bigram_counts_graph).reset_index()

    #bigram_df = pd.DataFrame.from_dict(bigram_counts_graph, orient='index').reset_index()
    bigram_df.columns = ['index','bigram','count']
    #bigram_df = pd.DataFrame(bigram_df, columns = ['bigram', 'count'])


    # ATTEMPT
    # Create dictionary of bigrams and their counts

    d = bigram_df.set_index('bigram').T.to_dict('records')

    # Create network plot 
    G = nx.Graph()

    type(ngram_frequency)

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    #G.add_node("china", weight=100)

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=1)

    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=16,
                     width=3,
                     edge_color='grey',
                     node_color='purple',
                     with_labels = False,
                     ax=ax)

    # Default values

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.04, value[1]+.01
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=13)
    
    plt.show()
    
%matplotlib auto
%matplotlib inline

chosen_cluster = tokens[4]
words_s = ['60','$']
for word in chosen_cluster:  # iterating on a copy since removing will mess things up
    if word in words_s:
        chosen_cluster.remove(word)

bigram_network(chosen_cluster)
plt.title("Finance and credit cluster for Uribe I")


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

Santos_II =  conpes_text[conpes_text['presidente']=='Juan Manuel Santos II']


# Start with one review:
text = str(Santos_II.texto)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()