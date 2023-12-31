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
######################################################

# My custom libraries
import personal_functions # see personal_functions.py of same dir = my personal functions list
import gzip # for unzipping the model file
import shutil # for unzipping the model file
from operator import itemgetter # To sort dictionary by key value
import sys # for sys.exit(INT)
import time # for time.sleep(INT)

######################################################

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


#####################################################

# Conserving the current dir before script was executed
# For returning to it at the script end
CURRENT_DIR_AT_THE_BEGGINING_OF_SCRIPT = os.getcwd()
# Changing dir to the script environment
PATH_OF_SCRIPT = os.path.realpath(__file__)
BASENAME_OF_SCRIPT = os.path.basename(PATH_OF_SCRIPT)
DIR_OF_SCRIPT = os.path.dirname(PATH_OF_SCRIPT)
# Changing dir to the script one
os.chdir(DIR_OF_SCRIPT)
# Getting one level up
os.chdir('..')
DIR_OF_PROJECT = os.getcwd()

#####################################################

MODEL = 'GoogleNews-vectors-negative300.bin'

# Manually downloaded the gzip file as other available are corrupted
# And as the official one on the Google Drive can not be download with
# GET request, only with POST request
# and no idea of the parameters required... 
path = DIR_OF_PROJECT + "/data" + "/" + MODEL + ".gz"

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

# Getting words of near context - see function above
show_words_of_near_context_from_string(model, "Germany")
show_words_of_near_context_from_string(model, "Annita_Kirsten")

# csv.DictReader create a dict class object with the keys from a file
# and then list() create a list object = list of dictionaries
countries = list(csv.DictReader(open('data/countries.csv')))

# random.sample returns a list of item from a list, tuple, string or set.
# Here : get a list of 40 countries names.
positive = [x['name'] for x in random.sample(countries, 40)]

# Getting a list of 5000 random words from the model
negative = random.sample(model.index_to_key, 5000)

# Creating tuples list in the format (country_name, 1)
list_of_tuples_from_positive = [(p, 1) for p in positive]
#personal_functions.printing_list_of_tuples(list_of_tuples_from_positive)
#print("\n\n\n") # Esthetic
# Creating tuples list in the format (random_word, 0)
list_of_tuples_from_negative = [(n, 0) for n in negative]
#personal_functions.printing_list_of_tuples(list_of_tuples_from_negative)
#print("\n\n\n") # Esthetic

# List concatenation
labelled = list_of_tuples_from_positive + list_of_tuples_from_negative
# labelled = [(p, 1) for p in positive] + [(n, 0) for n in negative]
#personal_functions.printing_list_of_tuples(labelled)
#print("\n\n\n") # Esthetic

# Shuffling the list
random.shuffle(labelled)

# numpy.asarray converts input
#
# list, list of tuples,
# tuples, tuples of tuples,
# tuples of list and ndarray
#
# to an array (to an 'ndarray')
# so X = a list of vector of numbers from the model for each word/country name = model[country_name] or [random_word]
X = np.asarray([model[w] for w, l in labelled])
# y = a list of numbers (0 or 1)
y = np.asarray([l for w, l in labelled])

# X.shape, y.shape = ndarray.shape
# = a tuple which show dimensions of the array (X and y)
#print("Dimensions of ndarray X : " + printing_tuple(X.shape))
#print("Dimensions of ndarray y : " + printing_tuple(y.shape))

# Training fraction = a percentage of the sample
TRAINING_FRACTION = 0.3
# Round to int the number of elements * the percentage
cut_off = int(TRAINING_FRACTION * len(labelled))

# svm = sklearn.svm (scikit learn)
# Wikipedia : "support vector machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis"
# Here it is precised the kernel type to be used in the algorithm
# The kernel matrix : an array of shape (n_samples, n_samples)
# svc takes two array as entries :
#    an array of the training samples
#    an array of class label (classification of the samples) = strings of INTs
# Here : creation of the model
# clf does not produce any printable result, it only creates a model with linear kernel (kind of model - there are other)
clf = svm.SVC(kernel='linear')

# fit(X, y[, sample_weight]) : Fit the SVM model according to the given training data.
# Returns: self object : Fitted estimator.
# Here, using the first INT elements of X and y
# Reminder :
#    X is the vector of numbers for each word taken from the model
#    y is a list of number 0 or 1
# So the clf.fit takes the vector representing a word (=X) and fits it to 0 or 1 (=y)
# And then after, it can makes prediction for other samples
# clf.fit does not produce any printable results, it only fits the model to the class labels in order to make prediction afterwards
clf.fit(X[:cut_off], y[:cut_off])

# res : 3528 elements consisting of values 0 or 1
# predict() : Perform classification on samples in X.
# y_predndarray of shape (n_samples,) : Class labels for samples in X.
# Here, try to predict the result for all elements after the INT first elements
# So will predict the class label (=classification from above y (=number 0/1) from the word vector (X) and produce a list of labels (0/1)
res = clf.predict(X[cut_off:]) 

# Here :
# zip will create tuples of (0/1, 0/1, (name, 0 / country,1)):
# res : the presumed result elements (number 0 (if name) or 1(if country)) = 3528 elements consisting of values 0 or 1
# the y elements (the real value : number 0 (if name) or 1(if country)) = 3528 elements consisting of values 0 or 1
# labelled[cut_off:] : 3528 elements consisting of tuples with (noun, 0) or (country, 1)
# Then, returns the country (tuple of (country,1))
# only if the prediction (res : 0/1) was different of the truth (y : 0/1)
# missed = list of tuples of format (country, 1)
missed = [country for (pred, truth, country) in 
 zip(res, y[cut_off:], labelled[cut_off:]) if pred != truth]

# Compute the percentage of right answers
percentage_of_failure = float(len(missed)) / len(res)
percentage_of_right_answers_float = 100 - 100 * percentage_of_failure 
# Format the float to max 4 digits, with max 2 decimals
# ':' takes the argument provided to 'format' function
pretty_percentage = "{:4.2f}".format(percentage_of_right_answers_float)
# Showing the result (percentage + failure list)
print("Percentage of right answers : " + pretty_percentage, "%", "\n", "List of tuples for which the model prediction was wrong : ", "\n", missed, "\n")

# Warning about this step because it takes time
# So show when it starts
print("\nGet the prediction (classification labels) for all words (vector) in the model.\nWarning : can take time depending on CPU/GPU...\n")
# Here : get the prediction (classification labels) for all words (vector) in the model
# Reminder :
# model = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)
# model = KeyedVectors object
# all_predictions = list of numbers (=labels) 0/1 (0 = word / 1 = country) 
all_predictions = clf.predict(model.vectors) # similar to 'res' list above
# And show when the above step ends
print("\nThe prediction succesfully ended !\n")

# Information great to know : how many words are there in the model ?
print("\nThere are " + str(len(all_predictions)) + " words in the word2vec used.\n")

# Getting a list of countries from the prediction model
res = []
# model.index_to_key = list of the model words [ word1, ..., wordn ]
# all_predictions = list of numbers 0/1 (0 if word, 1 if country)
for word, pred in zip(model.index_to_key, all_predictions):
    # if pred = 0 > false / pred <> 0 = true
    # so here, try to check if it is a country (if pred = 1)
    if pred:
        # If so, append it to the res list
        res.append(word)
        # And break if as soon as the res list contains 150 countries
        if len(res) == 150:
            break
        
# Showing N countries randomly from the country result list
N = 10 # 0 <= N <= 150
list_of_N_countries_predicted = random.sample(res, N)
personal_functions.printing_list_elements(list_of_N_countries_predicted)

# Reminder : countries = list(csv.DictReader(open('data/countries.csv')))
# = the list of dictionaries for all countries
# with each element in the format {name:value, cc:value, cc3:value }
# enumerate returns an iterator with index and element pairs from the original iterable
# By default, start index = 0
# So here, returns a new dictionary with the country name (key) and corresponding index (value) : { "Canada" : 0, ..., "Comoros" : 183 }
country_to_idx = {country['name']: idx for idx, country in enumerate(countries)}

# cnp.asarray : Convert the input to an array > returns an ndarray (country_vecs)
# 'c' represent each dictionary (one dict per country) in the format {name:value, cc:value, cc3:value }
# country_vecs get the vectors for each country name for the csv file
# by getting it from the model from the key value 'name' of country dict
country_vecs = np.asarray([model[c['name']] for c in countries])

# Showing the dimensions of the new array of countries vectors from the model
print("\nThe new array of countries vectors from the model has dimensions :")
personal_functions.printing_tuple(country_vecs.shape) # dimensions : ( 184, 300 ) = 184 countries and a vector of 300 numbers with each
print() # Esthetic

# np.dot : dot product or scalar product is an algebraic operation that takes two equal-length sequences of numbers, and returns a single number.
# country_to_idx['Canada'] : the index returned from countries
# country_vecs[country_to_idx['Canada']] : the vector of the same index - therefore, the vector for 'Canada'
# So here : scalar product of all countries vector and the Canada vector

print("\n\nDEBUG")
print("1. Show what is : country_vecs[country_to_idx['Canada']]")
print("2. Compare from the value of model['Canada'] to check that it is the same")
print("3. Check why to use the '.dot' to compute a scalar vector")
print("4. How is it done (mathematics) ?")
print("5. What does the np.argsort() function ?")
print("6. How is the np.argsort() processed ?")
print("7. What does the reversed function and what is the return value ?")
input("\nDEBUG DO THE ABOVE")

dists = np.dot(country_vecs, country_vecs[country_to_idx['Canada']])
for idx in reversed(np.argsort(dists)[-10:]):
    print(countries[idx]['name'], dists[idx])

input("DEBUG I AM HERE")
"""
    
def rank_countries(model, term, topn=10, field='name'):
    if not term in model:
        return []
    vec = model[term]
    dists = np.dot(country_vecs, vec)
    return [(countries[idx][field], float(dists[idx])) 
            for idx in reversed(np.argsort(dists)[-topn:])]

input("DEBUG MOVE THE ABOVE FUNCTION AT THE TOP OF SCRIPT WHEN UNDERSTOOD")

rank_countries(model, 'cricket')

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head()

def map_term(term):
    d = {k.upper(): v for k, v in rank_countries(model, term, topn=0, field='cc3')}
    world[term] = world['iso_a3'].map(d)
    world[term] /= world[term].max()
    world.dropna().plot(term, cmap='OrRd')

map_term('coffee')

map_term('cricket')

map_term('China')

map_term('vodka')

"""

input("\n\nDEBUG TO DO : RENAME SCRIPT DEPENDING OF FINAL CONTENT UNDERSTANDING")

# Getting back to dir before execution of script
os.chdir(CURRENT_DIR_AT_THE_BEGGINING_OF_SCRIPT)
print(f'DEBUG : END : PWD : {os.getcwd()}')

