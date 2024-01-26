from sklearn import svm
from keras.utils import get_file
import os
# Change locally from walyand to xorg because of hardware/software incompatibily issue
# Setting two env vars to get rid of the warning message : "Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway."
# Because current config is set to wayland windows manager (and not Xorg)
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
import gensim
import numpy as np
import random
import requests
import geopandas as gpd
from IPython.core.pylabtools import figsize
figsize(12, 8) # Set the default figure (for plots) size to be [sizex, sizey].
import csv

######################################################

# My custom libraries
import personal_functions # see personal_functions.py of same dir = my personal functions list
from operator import itemgetter # To sort dictionary by key value
import sys # for sys.exit(INT)
import time # for time.sleep(INT)
import subprocess # to execute shell command from script - here to open PDF files

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

def rank_countries(model, term, country_vecs, countries, topn=10, field='name', silent_true=False):
    """
    Returns a tuple with the content of the key value and the dot vector from the countries having the more in common with the term.
    """
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
    # Adapt the number of elements shown in the message below
    # If topn = 0, all the elements will be listed so take the len of the results list
    if topn == 0:
        NUMBER_OF_ELEMENTS_SHOWN = len(LIST_OF_TUPLES_RESULTS)
    else:
        NUMBER_OF_ELEMENTS_SHOWN = topn
    # Show results only if not silent mode (silent = false)- else only return them
    if not silent_true:
        print("\nThe " + str(NUMBER_OF_ELEMENTS_SHOWN) + " country/ies of context the closest to the term \"" + str(term) + "\" are :\n")
        personal_functions.printing_list_of_tuples(LIST_OF_TUPLES_RESULTS)
    # In all case (silent or not) > return the result
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

def map_term(model, term, country_vecs, countries, world, file_dir_and_prefix, matching_key):
    # Show a world map with color and intensity depending on how much the term is linked to the country (for every country in the world - if data is available for it).
    # Above parameter 'matching_key' = to adapt the matching key 'iso_a3' or 'ISO_A3' depending on the data file used (and therefore of the key name within)
    
    # rank_countries : return a tuple with the content of the key value and the dot vector from the countries having the more in common with the term.
    # Therefore :
    #    k = the key value (cc3 field = abbreviation of the country name) - set to upper letter to match the 'world' object which has ISO_A3 (upper) keys
    #    v = the dot product of this country with the term
    # So here, create a dictionary of UPPER CHAR cc3 field as a key and the dot product as the key value 
    d = {k.upper(): v for k, v in rank_countries(model, term, country_vecs, countries, topn=0, field='cc3', silent_true=True)}
    # map(function) : Apply a function to a Dataframe elementwise. -> the function returns the dot product for the key
    # So here : world[term] is an object representing the list of dot product between the term and each country (which is mapped through its cc3=iso_a3 key)
    world[term] = world[matching_key].map(d)
    """
    # Example of world[term] value:
    0      0.806491
    1      1.883944
    2      0.963322
    3      0.409483
    4      0.420392
             ...   
    172   -0.303006
    173   -0.089909
    174         NaN
    175         NaN
    176         NaN
    """
    # Here : scaling the data (=dividing by the max) before processing it (plot)
    # Why to scale : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
    # In short, gain of computation time and more stability in the model
    # Example of world[term].max() value : 1.9432250261306763
    world[term] /= world[term].max()
    
    # .dropna() : remove missing values
    # .plot : Generate a plot of a GeoDataFrame with matplotlib. 
    #     showing the plot of the map depending on the term and the countries linked to it
    # column : The name of the dataframe column, np.array, or pd.Series to be plotted.
    # cmap : str (default None) > The name of a colormap recognized by matplotlib.
    #    see the list of colormaps available here :
    #        https://matplotlib.org/stable/users/explain/colors/colormaps.html
    # figsize : Size of the resulting matplotlib.figure.Figure. If the argument axes is given explicitly, figsize is ignored. - Here figsize is defined on the top part of the script.
    # Reminder about figsize : comes from "from IPython.core.pylabtools import figsize"
    # So here : Return a plot with a world map with intensity varying on the dot product between the term and each country.
    PLOT_ABOUT_TERM = world.dropna().plot(column=term, cmap='Blues')
    
    # get_figure() : Return the Figure instance the artist belongs to.
    # savefig() : Save the current figure (with optional custom format)
    # Both function comes from 'matplotlib.figure.Figure' object
    # So here : save the world map into a PDF file
    PLOT_ABOUT_TERM.get_figure().savefig(file_dir_and_prefix + str(term) + ".pdf", format='pdf')

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
# so x = a list of vector of numbers from the model for each word/country name = model[country_name] or [random_word]
x = np.asarray([model[w] for w, l in labelled])
# y = a list of numbers (0 or 1)
y = np.asarray([l for w, l in labelled])

# x.shape, y.shape = ndarray.shape
# = a tuple which show dimensions of the array (x and y)
#print("Dimensions of ndarray x : " + printing_tuple(x.shape))
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

# fit(x, y[, sample_weight]) : Fit the SVM model according to the given training data.
# Returns: self object : Fitted estimator.
# Here, using the first INT elements of x and y
# Reminder :
#    x is the vector of numbers for each word taken from the model
#    y is a list of number 0 or 1
# So the clf.fit takes the vector representing a word (=x) and fits it to 0 or 1 (=y)
# And then after, it can makes prediction for other samples
# clf.fit does not produce any printable results, it only fits the model to the class labels in order to make prediction afterwards
clf.fit(x[:cut_off], y[:cut_off])

# res : 3528 elements consisting of values 0 or 1
# predict() : Perform classification on samples in x.
# y_predndarray of shape (n_samples,) : Class labels for samples in x.
# Here, try to predict the result for all elements after the INT first elements
# So will predict the class label (=classification from above y (=number 0/1) from the word vector (x) and produce a list of labels (0/1)
res = clf.predict(x[cut_off:]) 

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
# Adapting the 's' at the end of the noun if plural (just for the fun)
number_of_wrong_predictions = len(missed)
tuples_singular_or_plural = "tuple"
if number_of_wrong_predictions > 1:
    tuples_singular_or_plural += "s"    
# Showing the result (percentage of correct results + number of failure(s) + failure(s) list)
# Here I trained the syntax of the 'print' function output (with commas)
print("Percentage of right answers : " + pretty_percentage, "%", "\nList of", str(number_of_wrong_predictions), tuples_singular_or_plural, "for which the model prediction was wrong : ")
print(missed, "\n")

# Warning about this step because it takes time
# So show when it starts
print("\nGet the prediction (classification labels) for all words (vector) in the model.\nWarning : can take time depending on CPU/GPU...\n")
# Here : get the prediction (classification labels) for all words (vector) in the model
# Reminder :
# model = gensim.models.KeyedVectors.load_word2vec_format(unzipped_model_path, binary=True)
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

# Original website of world maps : https://www.naturalearthdata.com/downloads/
# Below url obtained with 'curl -v -L COPY_PASTED_LINK_FROM_WEBSITE' to show the final url redirected
WORLD_MAP_DATA_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

# Customized pretty dirname
WORLD_DATA_DIR_BASENAME = "world_map_data"
# Create the path name of future zipped (dir) file 
PATH_ZIPPED_WORLD_MAP_DATA = DATA_DIR + "/" + WORLD_DATA_DIR_BASENAME + ".zip"
# Download the data (dir) file from url
personal_functions.download_data_file_from_url(WORLD_MAP_DATA_URL, PATH_ZIPPED_WORLD_MAP_DATA)
# Unzip the dir if not done and get unzipped dir path
unzipped_world_map_dir = personal_functions.unzip_data_file(PATH_ZIPPED_WORLD_MAP_DATA, binary_true=False)

# Specify a dynamic data filename (with extension .shp) from the url - it will then be used below
# Get basename from url
# split from the right by cutting at the first separator '/' just once
# Results is a list of two element
# Then get element at index 1 = the last one
URL_BASENAME = WORLD_MAP_DATA_URL.rsplit('/', 1)[1]
# Remove extension
# Roughly the same as above this time, get the right element (index 0)
URL_BASENAME_WITHOUT_EXTENSION = URL_BASENAME.rsplit('.', 1)[0]
# Add extension .shp
DATA_FILENAME_BASENAME = URL_BASENAME_WITHOUT_EXTENSION + ".shp"

# Concatenate dir + data file basename to get data filename
PATH_OF_WORLD_MAP = unzipped_world_map_dir + "/" + DATA_FILENAME_BASENAME
# gpd = geopandas
# gpd.read_file : Returns a GeoDataFrame from a file or URL.
# So here : get the path from the file name, read it and return a GeoDataFrame representing the world.
# GeoDataFrame : a tabular data structure (pandas DataFrame - see below) that contains a column which contains a GeoSeries storing geometry.
# Reminder : a pandas.DataFrame is a Two-dimensional, size-mutable, potentially heterogeneous tabular data.





input("DEBUG : issue with the data file - trying to understand why - because with original data it works but not with my custom data...")
world = gpd.read_file(PATH_OF_WORLD_MAP)
# Defining the equivalent key of 'cc3' from the data file (can be 'iso_a3' or 'ISO_A3' depending on it)
matching_key = "ISO_A3"

# Original code below (but creates a warning so corrected above
#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Defining the equivalent key of 'cc3' from the data file (can be 'iso_a3' or 'ISO_A3' depending on it)
#matching_key = "iso_a3"





# Checking that the bounds from the model and from the GeoDataFrame (by default) are compatible
# GeoDataFrame.total_bounds : returns a tuple containing minx, miny, maxx, maxy values for the bounds of the series as a whole.
print(world.total_bounds) # [-180.          -90.          180.           83.63410065]

# GeoDataFrame.crs : The coordinate reference system (CRS) = see also below the EPSG
# From the official doc : It is important because the geometric shapes in a GeoSeries or GeoDataFrame object are simply a collection of coordinates in an arbitrary space.
# A CRS tells Python how those coordinates relate to places on the Earth.
# EPSG (wikipedia)
"""
EPSG Geodetic Parameter Dataset (also EPSG registry) is a public registry of geodetic datums, spatial reference systems, Earth ellipsoids, coordinate transformations and related units of measurement, originated by a member of the European Petroleum Survey Group (EPSG) in 1985. Each entity is assigned an EPSG code between 1024 and 32767,[1][2] along with a standard machine-readable well-known text (WKT) representation. The dataset is maintained by the IOGP Geomatics Committee.[3]
"""
# See also : https://spatialreference.org/ &  https://epsg.io/
print(world.crs) # EPSG:4326 = bounds [-180.          -90.          180.           90.] = ok
      
# Printing data sample from 'world' map data file
# GeoDataFrame.head([n]) : Return the first n rows.
"""
Output example of world.head() :

       pop_est  ...                                           geometry
0     889953.0  ...  MULTIPOLYGON (((180.00000 -16.06713, 180.00000...
1   58005463.0  ...  POLYGON ((33.90371 -0.95000, 34.07262 -1.05982...
2     603253.0  ...  POLYGON ((-8.66559 27.65643, -8.66512 27.58948...
3   37589262.0  ...  MULTIPOLYGON (((-122.84000 49.00000, -122.9742...
4  328239523.0  ...  MULTIPOLYGON (((-122.84000 49.00000, -120.0000...

[5 rows x 6 columns]

"""
print(world.head())
print() # Esthetic

# Set where the PDF of plot results will be saved
pdf_files_dir = DATA_DIR + "/" + "world_map_pdf"
# Create the dir if not yet done
personal_functions.create_dir_if_not_existing(pdf_files_dir)
# Customized file prefix if wished - else set to ''
file_prefix = "map_world_"
file_dir_and_prefix = pdf_files_dir + "/" + file_prefix
# List of words for which to process a PDF
WORDS_LIST = [ 'coffee', 'cricket', 'China', 'vodka', 'Pablo' ]
for i in range(len(WORDS_LIST)):
    current_word = WORDS_LIST[i]
    # Saving several maps with country more linked to the term colored more intensely
    map_term(model, current_word, country_vecs, countries, world, file_dir_and_prefix, matching_key)
    path_filename_world_map_with_word = file_dir_and_prefix + current_word + ".pdf"
    # Showing the pdf files by opening it with default app (from xdg-open)
    subprocess.call(["xdg-open", path_filename_world_map_with_word])
    
# Getting back to the dir before execution of script
os.chdir(CURRENT_DIR_AT_THE_BEGGINING_OF_SCRIPT)

print("\n\nTO DO")
print("To better understand the model : Run the original model to train from words - see favorite : website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb")
print()
input("\nDEBUG : above things to be processed before leaving the script\n")

