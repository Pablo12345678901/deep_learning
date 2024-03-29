import os
import sys # For sys.exit(EXIT_CODE)
from keras.utils import get_file
import gensim
import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # set for the interractive showing of plots
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(10, 10)
from sklearn.manifold import TSNE
import json
from collections import Counter
from itertools import chain
# To unzip the binary model file
import shutil
import gzip
# To randomly shuffle a list
import random

import personal_functions # my personal functions

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
DIR_OF_PROJECT = os.getcwd()

################################################################
print("\n\nFirst part of the script: \nLoading the word2vec model from a gzip file.\n\n")
################################################################

MODEL = 'GoogleNews-vectors-negative300.bin'

# Manually downloaded the gzip file as other available are corrupted
# And as the official one on the Google Drive can not be download with
# GET request, only with POST request
# and no idea of the parameters required...
DATA_DIR = DIR_OF_PROJECT + "/" + "data" 
path = DATA_DIR + "/" + MODEL + ".gz"

# Create data dir if not yet existing
personal_functions.create_dir_if_not_existing(DATA_DIR)

# Unzipping the model dir (if not yet done) and getting its name
unzipped_model_dir = personal_functions.unzip_data_file(path, binary_true=True)
# Get the name of the model unzipped file
unzipped_model_path = unzipped_model_dir + "/" + MODEL
   
# Model loading
print("\nLoading the word2vec model...")
model = gensim.models.KeyedVectors.load_word2vec_format(unzipped_model_path, binary=True)
print("Finished to load the word2vec model !\n")

############################################################################
print("\n\nSecond part of script : \nShowing that word2vec model is able to give words synonyms and near context words.\n\n")
############################################################################

# Get a list of tuples with two elements
# the similar word and the computed value (number) for each
similar_terms_list_of_tuple = model.most_similar(positive=['espresso'])
# Get the first element of each tuple only (the word) into a list
similar_terms_list=[similar_terms_tuples[0] for similar_terms_tuples in similar_terms_list_of_tuple ]
# Join the list to a string
similar_terms_string = ', '.join(similar_terms_list) # return error expected a string but it is a 'tuple'
print("\nSynonyms of espresso : " + similar_terms_string + ".\n")


def A_is_to_B_as_C_is_to(a, b, c, topn=1):
    # map : return a map object (iterable) after aplying a function
    # lambda : anonymous function for oneshot use
    # lambda here : from 'x' return 'x' if the type of x is a list,
    # else return a list composed of only one element 'x'
    # So, a, b, c then become a list composed of [a], [b] and [c]
    a, b, c = map(lambda x:x if type(x) == list else [x], (a, b, c))
    # Get the most similar word by computing :
    # the similarity between 'b'+'c' and substratcing the element 'a'
    res = model.most_similar(positive=b + c, negative=a, topn=topn)
    if len(res):
        # If the number of elements required is 1, returns his value (string)
        if topn == 1:
            return res[0][0]
        # else return a lists of N elements 
        else:
            return [x[0] for x in res]
    else:
        return None
    
print("\nIf a man is a king, then a woman is a : %s" % (A_is_to_B_as_C_is_to('man', 'woman', 'king')) + " !\n")

print("\n") # Esthetic
for country in 'Italy', 'France', 'India', 'China':
    print('%s is the capital of %s' % 
          (A_is_to_B_as_C_is_to('Germany', 'Berlin', country), country))
print("\n") # Esthetic

print("\n") # Esthetic
for company in 'Google', 'IBM', 'Boeing', 'Microsoft', 'Samsung':
    # Here the function will map the two lists composed of the company and its product
    # To generate three other products name from the company. 
    products = A_is_to_B_as_C_is_to(
        ['Starbucks', 'Apple'], 
        ['Starbucks_coffee', 'iPhone'], 
        company, topn=3)
    print('%s -> %s' % 
          (company, ', '.join(products)))
print("\n") # Esthetic

################################################################
print("\n\nThird part of the script : \nShowing the that the word2vec model is able to recognize words of thematic similar words (with a plot regrouping the words).\n\n")
################################################################

# List creation
beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
countries = ['Italy', 'Germany', 'Russia', 'France', 'USA', 'India']
sports = ['soccer', 'handball', 'hockey', 'cycling', 'basketball', 'cricket']
# List concatenation
items = beverages + countries + sports

def list_printing(list_to_print):
    print("\nList content :")
    for i in range(len(list_to_print)):
        print(str(list_to_print[i])) # conversion to string, just in case it is used with numbers.
    # Or
    # [ print(str(element)) for element in list_to_print]
    print("\n") # Esthetic
    
# Print list content before shuffling
print("\nPrinting list content BEFORE shuffling\n")
list_printing(items)
# Shuffling list
print("\nShuffling elements of list...\n")
random.shuffle(items)
# Print list content after shuffling
print("\nPrinting list content AFTER shuffling\n")
list_printing(items)

# Print lenght
items_list_lenght = len(items)
print("\nLen of the items list : " + str(items_list_lenght) + ".\n")

# List creation based on other list
# For each item in item list (=items),
# if the item is in model,
# then create a tuple composed of :
#    the item and
#    of the vector of this item from the model. 
item_vectors = [(item, model[item]) 
                    for item in items
                    if item in model] # [ a(b) for y in z in y in ] 
# Print lenght
item_vectors_list_lenght = len(item_vectors)
print("\nLen of the items vector list : " + str(item_vectors_list_lenght) + ".\n")

# Convert the tuples composed of the first element of the tuple item_vectors
# that is to say the tuples 'model[item]'
# to an array.
# A base class ndarray is returned
vectors = np.asarray([x[1] for x in item_vectors])

def show_vectors_content(vectors):
    # Getting the lenght
    vectors_lenght = len(vectors)
    print("\nLen of the vectors list : " + str(vectors_lenght) + ".\n")
    # Printing the content of each vector within 'vectors'
    print("\nShowing content of 'vectors' element")
    for i in range(vectors_lenght):
        print("\nElement " + str(i) + "\n")
        print(vectors[i])
    print("\nEnd of showing...\n")
    return None

# Uncomment the below line if you are curious about the 'vectors' data. 
#show_vectors_content(vectors)

# Get the norm for each of the vectors list
lengths = np.linalg.norm(vectors, axis=1)

norm_vectors = (vectors.T / lengths).T
# vectors.T : np.ndarray class - attrbute '.T' : View of the transposed array

#T-distributed stochastic neighbor embedding (t-SNE) is
# a technique that helps users visualize high-dimensional data sets.
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
tsne = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(norm_vectors)
# n_components : dimension number
# perplexity : complex concept > see official documentation.
# verbose : verbosity level

# Get the coordinates
x=tsne[:,0] 
y=tsne[:,1]
# Slicing theory (but above, only the slice ':' is used = the whole array
# a[:]           # a copy of the whole array
# a[start:]      # items start through the rest of the array
# a[:stop]       # items from the beginning through stop-1
# a[start:stop]  # items start through stop-1

# Create a plot and return the figure (fig) and the axis (ax)
fig, ax = plt.subplots()
# Filling the data into the axis
ax.scatter(x, y)

# The zip() function returns a zip object, which is an iterator of tuples
# where the first item in each passed iterator is paired together,
# and then the second item in each passed iterator are paired together etc.
# Zip combine lists elements (first one with first one) of same list sizes.
# Here zip combine the word (and number data), the x and y coordinates 
for item, x1, y1 in zip(item_vectors, x, y):
    # Get the name (first element of tuple) and its coordinates
    # And write them into the plot
    ax.annotate(item[0], (x1, y1), size=14)

# Show plot graphic
# This plot shows the groups between words on similar thematic.
plt.show()

# Getting back to the dir before execution of script
os.chdir(CURRENT_DIR_AT_THE_BEGGINING_OF_SCRIPT)

# To go for an extra mile
print("\n\nTo better understand the word2vec model, you can run the original model to train from words - see favorite : website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb\n\n")

# End message and exit without error
print("\nScript finished !\n")
sys.exit(0)

