from sklearn import svm
from keras.utils import get_file
import os
import gensim
import numpy as np
import random
import requests
import geopandas as gpd
from IPython.core.pylabtools import figsize
# figsize(12, 8) # don't know yet if it is important - was on the top on the Notebook
import csv
from operator import itemgetter # To sort dictionary


MODEL = 'GoogleNews-vectors-negative300.bin'

# Manually downloaded the gzip file as other available are corrupted
# And as the official one on the Google Drive can not be download with
# GET request, only with POST request
# and no idea of the parameters required... 
path = "/home/incognito/Desktop/developpement/deep_learning/dev_and_tests_from_deep_learning_cookbook_examples/data" + "/" + MODEL + ".gz"

# Creating dir if not existing
if not os.path.isdir('data'):
    os.mkdir('data')

# Get name of future unzipped file
unzipped = os.path.join('data', MODEL)

# If future unzipped file is not existing yet - unzipping from file downloaded above
if not os.path.isfile(unzipped):
    with gzip.open(path, 'rb') as file_in:
        with open(unzipped, 'wb') as file_out:
            print("\nUnzipping the binary file located on path \"" + path + "\"")
            shutil.copyfileobj(file_in, file_out)
            print("Binary file obtained after successfully unzipping !\n")
        
# Model loading
print("\nLoading the word2vec model...")
model = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)
print("Finished to load the word2vec model !\n")

def show_words_of_near_context_from_string(model, a_string):
    # Get a list of tuples with two elements
    # the similar word and the computed value (number) for each
    similar_terms_list_of_tuple = model.most_similar(positive=[a_string])
    # Get the first element of each tuple only (the word) into a list
    similar_terms_list=[similar_terms_tuples[0] for similar_terms_tuples in similar_terms_list_of_tuple ]
    # Join the list to a string
    similar_terms_string = ', '.join(similar_terms_list) # return error expected a string but it is a 'tuple'
    print("\nWords in the near context of " + a_string + " : " + similar_terms_string + ".\n")
    return None

show_words_of_near_context_from_string(model, "Germany")
# model.most_similar(positive=['Germany'])
show_words_of_near_context_from_string(model, "Annita_Kirsten")
#model.most_similar(positive=['Annita_Kirsten'])


# csv.DictReader create a dict class object with the keys from a file
# and then list() create a list object = list of dictionaries
countries = list(csv.DictReader(open('data/countries.csv')))

# Below : customs list of dict functions to print them in a pretty way
def print_dictionary_elements(dictionary, index):
    print(str(index))
    for key, value in dictionary.items():
        print(key + " : " + value)
    print() # Esthetic

def print_list_of_dictionaries(list_of_dictionaries):
    for i in range(len(list_of_dictionaries)):
        print_dictionary_elements(list_of_dictionaries[i], i)

#print_list_of_dictionaries(countries)
        
def print_sorted_list_of_dictionaries_by_a_key_value(list_of_dictionaries, value):
    newlist = sorted(list_of_dictionaries, key=itemgetter(value), reverse=False)
    print("\nList of dictionaries sorted by the value of the key " + value)
    for i in range(len(newlist)):
        print_dictionary_elements(newlist[i], i)

#print_sorted_list_of_dictionaries_by_a_key_value(countries, "name")

# Show the first ten items of the dictionary (stocked wihtin a list range)
# But as it is a dictionary, the order is linked to the hash and seems 'random'
#print(countries[:10]) 

def printing_list_elements(list):
    list_lenght = len(list)
    print("\nPrinting " + str(list_lenght) + " elements of the list :")
    for i in range(list_lenght):
        print(str(i) + " : " + list[i])
    print() # Esthetic

# random.sample returns a list of item from a list, tuple, string or set.
# Return a list of 40 country names.
positive = [x['name'] for x in random.sample(countries, 40)]
#printing_list_elements(positive)

# Getting 5000 random words from the model
negative = random.sample(model.index_to_key, 5000)
#printing_list_elements(negative)

### I AM HERE ########################

"""

labelled = [(p, 1) for p in positive] + [(n, 0) for n in negative]
random.shuffle(labelled)
X = np.asarray([model[w] for w, l in labelled])
y = np.asarray([l for w, l in labelled])
X.shape, y.shape

TRAINING_FRACTION = 0.3
cut_off = int(TRAINING_FRACTION * len(labelled))
clf = svm.SVC(kernel='linear')
clf.fit(X[:cut_off], y[:cut_off])

res = clf.predict(X[cut_off:])

missed = [country for (pred, truth, country) in 
 zip(res, y[cut_off:], labelled[cut_off:]) if pred != truth]

100 - 100 * float(len(missed)) / len(res), missed

# BUG HERE : AttributeError: 'KeyedVectors' object has no attribute 'syn0'
print("128 BUG HERE")
all_predictions = clf.predict(model.syn0)

res = []
for word, pred in zip(model.index2word, all_predictions):
    if pred:
        res.append(word)
        if len(res) == 150:
            break
random.sample(res, 10)

country_to_idx = {country['name']: idx for idx, country in enumerate(countries)}
country_vecs = np.asarray([model[c['name']] for c in countries])
country_vecs.shape

dists = np.dot(country_vecs, country_vecs[country_to_idx['Canada']])
for idx in reversed(np.argsort(dists)[-10:]):
    print(countries[idx]['name'], dists[idx])

def rank_countries(term, topn=10, field='name'):
    if not term in model:
        return []
    vec = model[term]
    dists = np.dot(country_vecs, vec)
    return [(countries[idx][field], float(dists[idx])) 
            for idx in reversed(np.argsort(dists)[-topn:])]

rank_countries('cricket')

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head()

def map_term(term):
    d = {k.upper(): v for k, v in rank_countries(term, topn=0, field='cc3')}
    world[term] = world['iso_a3'].map(d)
    world[term] /= world[term].max()
    world.dropna().plot(term, cmap='OrRd')

map_term('coffee')

map_term('cricket')

map_term('China')

map_term('vodka')

"""
