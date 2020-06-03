# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:38:21 2019

@author: User
"""

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import nltk
from collections import Counter
from nltk.util import ngrams
import networkx as nx
from nltk import bigrams
import itertools
import collections

#nltk.download('maxent_ne_chunker')
#nltk.download('words')

# GRAPHS USING THE TITLES
# Step 2
with open('C:\\Users\\User\\Desktop\\DS4A_workspace\\Final Project\\DS4A-Final_Project-Team_28\\Personal\\Camilo\\titles_clean', 'rb') as titles_2_clean_file:
 
    # Step 3
    titles_2_clean = pickle.load(titles_2_clean_file)
 
    # After config_dictionary is read from file
    print(titles_2_clean)
    

merged_titles = ' '.join(titles_2_clean)

# WORDCLOUD
wordcloud = WordCloud(width=1000, height=800, background_color="black").generate(merged_titles)

# Display the generated image:
# the matplotlib way:
%matplotlib auto

figure(num=None, figsize=(10, 10), dpi=1000, facecolor='w', edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off");
plt.show()


# WORD COUNT
tokens = nltk.tokenize.word_tokenize(merged_titles)
tokens_frequency = nltk.FreqDist(tokens)
#tokens_frequency.plot(30,cumulative=False)

tokens_frequency = dict(tokens_frequency.copy())
tokens_table = pd.DataFrame.from_dict(tokens_frequency, orient='Index', columns = ['frequency'])
tokens_table.reset_index(level=0, inplace=True)
# Order the table
tokens_table = tokens_table.sort_values(by=['frequency'], ascending = False)

# Barplot of most common words in all titles
sns.barplot("index", "frequency", data=tokens_table[:10], palette="Blues_d").set_title("Top 10 Words")


bg_dict = {('best', 'price'): 95, ('price', 'range'): 190, ('range', 'got'): 5, ('got', 'diwali'): 2, ('diwali', 'sale'): 2, ('sale', 'simply'): 1,
       ('simply', 'amazed'): 1, ('amazed', 'performance'): 1, ('performance', 'camera'): 30, ('camera', 'clarity'): 35, ('clarity', 'device'): 1,
       ('device', 'speed'): 1, ('speed', 'looks'): 1, ('looks', 'display'): 1, ('display', 'everything'): 2, ('everything', 'nice'): 5, ('nice', 'heats'): 2, ('heats', 'lot'): 14,
       ('lot', 'u'): 2, ('u', 'using'): 3, ('using', 'months'): 20, ('months', 'no'): 10, ('no', 'problems'): 8, ('problems', 'whatsoever'): 1, ('whatsoever', 'great'): 1}

bg_dict_sorted = sorted(bg_dict.items(), key=lambda kv: kv[1], reverse=True)
bg, counts = list(zip(*bg_dict_sorted))
bg_str = list(map(lambda x: '-'.join(x), bg))
sns.barplot(bg_str, counts, orient = 'h')

# WORD NETWORK OF BIGRAMS
ngram = list(ngrams(tokens, 2))
ngram_frequency = nltk.FreqDist(ngram)
#ngram_frequency.plot(30,cumulative=False)

ngram_frequency = dict(ngram_frequency.copy())
ngram_table = pd.DataFrame.from_dict(ngram_frequency, orient='Index', columns = ['frequency'])
ngram_table.reset_index(level=0, inplace=True)
# Order the table
ngram_table = ngram_table.sort_values(by=['frequency'], ascending = False)

terms_bigram = [list(bigrams(tokens))]
bigrams = list(ngrams(tokens, 2))
bigrams = list(itertools.chain(*terms_bigram))

bigram_counts = collections.Counter(bigrams)
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
    x, y = value[0]+.10, value[1]+.01
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
    
plt.show()
