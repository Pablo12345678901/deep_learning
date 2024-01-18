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

def rank_countries(model, term, country_vecs, countries, topn=10, field='name'):
    # Above argument 'country_vecs' = the vectors from the country names (in turn from the CSV)
    # If the word is not in the model, return an empty list
    if not term in model:
        print("\nERROR : the term \"" + str(term) + "\" is not in the model.\n")
        return []
    # Get the word vector from the model
    vec = model[term]
    # dot product between the country_vecs and the vec from the model based on the word
    dists = np.dot(country_vecs, vec)
    # Get the 'topn' countries the closest to the 'term' passed to the function by sorting their vectors (=np.argsort) from the smallest to the biggest, then taking only the last 'topn' elements (the 'topn' bigger vectors), then reversed their order (function 'reversed') to get the closest country first (= their indexes : idx)...
    # and lastly creating a tuple containing :
    #    - the countries key ('field') value (default to 'name') from the list of dictionary 'countries'
    #    - the dot product for this country
    LIST_OF_TUPLES_RESULTS = [(countries[idx][field], float(dists[idx]))
                      for idx in reversed(np.argsort(dists)[-topn:])]
    print("\nThe " + str(topn) + " country/ies of context the closest to the term \"" + str(term) + "\" are :\n")
    personal_functions.printing_list_of_tuples(LIST_OF_TUPLES_RESULTS)
    return LIST_OF_TUPLES_RESULTS

def show_range_of_vector_element_for_each_country_and_for_all_of_them(countries, model):
    # Reminder about 'countries' element :
    # countries = list(csv.DictReader(open('data/countries.csv')))
    # csv.DictReader create a dict class object with the keys from a file
    # and then list() create a list object = list of dictionaries
    country_list = [ c['name'] for c in countries ]
    # Show country list
    personal_functions.printing_list_elements(country_list)
    # Declare the general max and min outside of the loop because needed after it
    general_min = 0.
    general_max = 0.
    # Getting the min and max values from each country vector + printing it 
    for i in range(len(country_list)):
        # Printing the country name
        print(str(i) + " - country name : " + str(country_list[i]))
        # Getting the country vector from the model
        country_vector = [model[country_list[i]]] # [array([ 0... , ... , -0... ], dtype=float32)]
        country_vector_clean = country_vector[0] # [ 0... , ... , -0... ]
        # Setting an initial value for the max and min value of the country vector
        max_from_vector = country_vector_clean[0]
        min_from_vector = country_vector_clean[0]
        # Setting a value to the general min and max - only the first time
        if i == 0:
            general_max = max_from_vector
            general_min = min_from_vector
        # Check if current value is above max or below min and update max and min values
        for index in range(len(country_vector_clean)):
            if country_vector_clean[index] > max_from_vector:
                max_from_vector = country_vector_clean[index]
            if country_vector_clean[index] < min_from_vector:
                min_from_vector = country_vector_clean[index]
        # Print the results for each country
        print("Range from min " + str(min_from_vector) + " to max " + str(max_from_vector) + ".")
        # Update the general min and max values depending on the current country values
        if max_from_vector > general_max:
            general_max = max_from_vector
        if  min_from_vector < general_min:
            general_min = min_from_vector
    # Printing the general max & min values
    print("\nAll the values are between min : " + str(general_min) + " and max : " + str(general_max) + ".\n")
    # No value returned - only for showing
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

# Showing ranges of vectors values for each country
# just for showing that they are included in range [ -1, 1]
show_range_of_vector_element_for_each_country_and_for_all_of_them(countries, model)

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
# Theory : https://en.wikipedia.org/wiki/Dot_product
# And in context of AI : https://stats.stackexchange.com/questions/291680/can-any-one-explain-why-dot-product-is-used-in-neural-network-and-what-is-the-in
# To summarize : use dot product to check if two vectors are aligned or not and for how much. The bigger result in a list of results of dot product between a vector and another list of vector will be the more aligned vector. Which means that they are the more similar : same size and direction (angle) in a Z dimensions space.
# a = [ 1, 3, -5 ]
# b = [ 4, -2, -1 ]
# a DOT b = (1 * 4) + ( 3 * -2) + (-5 * -1)
# = 4 -6 + 5 = 3
# = a * transposed_matrix(b)
# country_to_idx['Canada'] : the index returned from countries
# country_vecs[country_to_idx['Canada']] : the vector of the same index - therefore, the vector for 'Canada', equivalent to the vector returned by : model['Canada']
# So here : dists = scalar product of all countries vectors and the Canada vector = to see how near they are.
dists = np.dot(country_vecs, country_vecs[country_to_idx['Canada']]) # returns : [ [ 7.544024 ... 1.7089474 ] = list of scalar products of each country with Canada

# Printing the 10 countries having the more in common with 'Canada' according to the model
print("\nPrinting 10 countries which vectors are the closest to the Canada one according to the word2vec model.\n\nSo they should have more in common (culture, nature, ...)  than compared to other countries.\n")
# np.argsort : Returns an ndarray containing the indices of an array that would be sorted (from the lowest value = 0 to the biggest = index max) along an axis (by default = -1 = the latest axis)
# [-10:] = the last 10 elements of the list = the 10 biggest value for scalar vector = (ex:) [ 23  37  41  17 125 100 134  27 162   0]
# reversed() : Return a reverse iterator : An object representing a stream of data. Repeated calls to the iterator’s __next__() method (or passing it to the built-in function next()) return successive items in the stream. When no more data are available a StopIteration exception is raised instead.
# So here : idx gets the index of the 10 last sorted elements of the list 'dists' containing the scalar product (of each country and 'Canada' for each one) and take them from the last one (the biggest number) to the first one (the 10th bigger number) element of the dists list.
for idx in reversed(np.argsort(dists)[-10:]):
    print("Country : " + countries[idx]['name'], " / Dot product : ", dists[idx])
print() # Esthetic

# Show the 10 countries the nearest to the 'cricket' word
rank_countries(model, 'cricket', country_vecs, countries)




input("DEBUG I AM HERE")
"""

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head()

def map_term(term):
    d = {k.upper(): v for k, v in rank_countries(model, term, country_vecs,countries, topn=0, field='cc3')}
    world[term] = world['iso_a3'].map(d)
    world[term] /= world[term].max()
    world.dropna().plot(term, cmap='OrRd')

map_term('coffee')

map_term('cricket')

map_term('China')

map_term('vodka')



# TO DO
input("\n\nDEBUG TO DO : RENAME SCRIPT DEPENDING OF FINAL CONTENT UNDERSTANDING")
input("\n\nTo do to better understand the model : Run the original model to train from words - see favorite : website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb \n\n")

"""

# Getting back to the dir before execution of script
os.chdir(CURRENT_DIR_AT_THE_BEGGINING_OF_SCRIPT)

