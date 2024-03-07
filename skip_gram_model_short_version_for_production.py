import io # Core tools for working with streams
import re # Regex operations
import string
import tqdm # Shows progress bar in Python

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# My custom imports
import personal_functions
import sys

############ OPTIONS MANAGEMENT ###########################

# By default, print additional output messages to help the script understanding
# Choose to print or not output messages + values at several steps
FLAG_SILENT_TRUE = False

# Manage options provided to the script
MAXIMUM_ARGUMENT_NUMBER = 1

# Compute number of arguments provided to the script
NUMBER_OF_PROVIDED_ARGUMENTS = len(sys.argv) - 1 # -1 because arg[0] = script name

# Check the argument(s) provided
# Reminder :
#    sys.argv[0] = name of the script
#    sys.argv[1] = first argument ... and so on
if NUMBER_OF_PROVIDED_ARGUMENTS == MAXIMUM_ARGUMENT_NUMBER:
    ARGUMENT_PROVIDED = str(sys.argv[1])
    # Set the flag of silent/verbose depending on the argument provided
    if ARGUMENT_PROVIDED == "--silent": 
        FLAG_SILENT_TRUE = True
    elif ARGUMENT_PROVIDED == "--verbose":
        FLAG_SILENT_TRUE = False
    else:
        # If the argument is not the expected one, exit with error
        print("\nERROR : unknown argument provided : " + ARGUMENT_PROVIDED + ".\n")
        sys.exit(1)
elif NUMBER_OF_PROVIDED_ARGUMENTS > MAXIMUM_ARGUMENT_NUMBER:
    # If too many arguments were provided, exit with error
    print("\nERROR : too many arguments provided : " + str(sys.argv[1:]) + ".\n")
    sys.exit(2)

############ FUNCTIONS LIST ######################

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size (and seed = random seed).
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):

    if not FLAG_SILENT_TRUE:
        print("Generate training data with those parameters :")
        print("sequences : ", sequences)
        print("window_size :", window_size)
        print("num_ns :", num_ns)
        print("vocab_size :", vocab_size)
        print("seed :", seed)
        print() # Esthetic
        
    # Define the lists that the function will fill with elements of each training example.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    # tf.keras.preprocessing.sequence.make_sampling_table : generate a word-frequency rank based probabilistic sampling table.
    # sampling_table[i] is the probability of sampling the word i-th most common word in a dataset (more common words should be sampled less frequently, for balance).
    # First index = most common word SO should be sampled 1/ratio to balance it and it is the number that sampling_table[i] returns.
    # The sampling probabilities are generated according to the sampling distribution used in word2vec.
    # Will be used by the 'tf.keras.preprocessing.sequence.skipgrams' function
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
  
    # Show the sampling probabilities table.
    if not FLAG_SILENT_TRUE:
        print("The sampling table is equivalent to ")
        personal_functions.printing_list_elements(sampling_table)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Show the step
        if not FLAG_SILENT_TRUE:
            print("Generate positive skip-gram pairs for the sequence (sentence) " , sequence)

        print("DEBUG 86 WHY NO VALUE FOR POSITIVE SKIP GRAMS")
            
        # Generate positive skip-gram pairs for the sequence
        # tf.keras.preprocessing.sequence.skipgrams : returns 'couples', 'labels' (here ignored because no negative samples): where couples are int pairs and labels are either 0 or 1. 
        positive_skip_grams, sequence_labels = tf.keras.preprocessing.sequence.skipgrams(
              sequence,
              vocabulary_size=vocab_size,
              sampling_table=sampling_table,
              window_size=window_size,
              negative_samples=0) # No negative samples here so the labels returned can be ignored.

    # Iterate over each positive skip-gram pair to produce training examples with a positive context word and negative samples.
    # Reminder : target_word = current word & context_word = word found around the current word
    for target_word, context_word in positive_skip_grams:
        
        # tf.expand_dims(input, axis, name=None) : Returns a tensor with a length 1 axis inserted at index axis.
        # Here : input = tf.constant & axis = 1
        # tf.constant : Creates a constant tensor from a tensor-like object.
        # So here : context_class is a tensor of 1 axis composed with the tensor constant equivalent to the context word
        context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)

        # tf.random.log_uniform_candidate_sampler : Returns :
        # -> sampled_candidates : A tensor of type int64 and shape [num_sampled]. The sampled classes. As noted above, sampled_candidates may overlap with true classes.
        # -> true_expected_count : A tensor of type float. Same shape as true_classes. The expected counts under the sampling distribution of each of true_classes.
        # -> sampled_expected_count : A tensor of type float. Same shape as sampled_candidates. The expected counts under the sampling distribution of each of sampled_candidates.
        # Here : true_expected_count & sampled_expected_count are ignored.
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling"
        )
        
        # Build context and label vectors (for one target word)
        # tf.concat(values, axis, name='concat')  : Concatenates tensors along one dimension.
        #
        # tf.squeeze(input, axis=None, name=None) : Removes dimensions of size 1 from the shape of a tensor.
        context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates],    0)

        print("DEBUG WHY NO CONTEXT ?")
        print(context)
        # Create a label list with one '1' and 0's = [ 1 0 ... 0 ]
        label = tf.constant([1] + [0]*num_ns, dtype="int64")
      
        # Append each element from the training example to global lists.
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

    return targets, contexts, labels

# Lowercase all data as well as removing all punctuation
def custom_standardization(input_data):
    # Lowercase all data
    lowercase = tf.strings.lower(input_data)
    # tf.strings.regex_replace(input, pattern, rewrite, replace_global=True, name=None) : Replace elements of input matching regex pattern with rewrite.
    # [%s] : to match any of the char within the string '%s'
    #  re.escape(pattern) : Escape special characters in pattern.
    # string.punctuation : String of ASCII characters which are considered punctuation characters in the C locale: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
    # Here, replace all strings of punctuation by '' = removal
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

######### DESCRIPTION OF FILE ################

if not FLAG_SILENT_TRUE:
    print("\nThis code is inspired from the website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb \nIt shows how to build a 'Continuous skip-gram model' which is able to predict words within a certain range before and after the current word in the same sentence. \nWhile a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context of target_word.\n")
    
############## MAIN CODE  ####################

# Define some data : 
# tf.data : an API that enables to build complex input pipelines from simple, reusable pieces.
# tf.data.AUTOTUNE : use tensorflow capacity to compute the processing time at each step while processing the input in order to optimize the pipeline while using this output (providing a better CPU usage) - more fluent.
# See : https://www.tensorflow.org/api_docs/python/tf/data#AUTOTUNE
# See : https://stackoverflow.com/questions/56613155/tensorflow-tf-data-autotune
AUTOTUNE = tf.data.AUTOTUNE # = -1
batch_size = 1024
vocab_size = 4 * batch_size # vocabulary size
sequence_length = 10 # number of words in a sequence

FILENAME = 'shakespeare.txt'
DOWNLOAD_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

# tf.keras.utils.get_file() : Downloads a file from a URL if it not already in the cache.
file_data_in_cache = tf.keras.utils.get_file(FILENAME, DOWNLOAD_URL)

# 
with open(file_data_in_cache) as text_data_file:
    # splitlines : split a text by lines with the ('\n')
    lines = text_data_file.read().splitlines()

    # Print the first X lines
    if not FLAG_SILENT_TRUE:
        NUMBER_OF_LINES_TO_PRINT = 20
        print("First " + str(NUMBER_OF_LINES_TO_PRINT) + " lines of the text : ")
        for i in range(0, NUMBER_OF_LINES_TO_PRINT):
            print(str(i) + ":" + str(lines[i]))
        print() # Esthetic

# Get the dataset from all non-empty lines
"""
tf.data.TextLineDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None,
    name=None
) : Creates a Dataset comprising lines from one or more text files. Inherits From: Dataset
"""
# Dataset.filter(predicate, name=None) -> 'DatasetV2' : Filters this dataset according to predicate.
# tf.cast(x, dtype, name=None) : cast a tensor to a new type.
# So here, returns only the lines that are not empty. Because lenght(0) = 0 = False & lenght(not_0) = True
text_ds = tf.data.TextLineDataset(file_data_in_cache).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Printing details about the variable
if not FLAG_SILENT_TRUE:
    print("'text_ds' (text dataset) : ")
    personal_functions.print_variable_information(text_ds, "text_ds")
    print() # Esthetic

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the same length.
# layers = tensorflow.keras.layers
"""
tf.keras.layers.TextVectorization(
    max_tokens=None,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding='utf-8',
    **kwargs
) : A preprocessing layer which maps text features to integer sequences. Inherits From: PreprocessingLayer, Layer, Module
"""
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization, # use the custom function to lowerxcase as well as remove all punctuation chars. 
    max_tokens=vocab_size,
    output_mode='int', # convert words to int
    output_sequence_length=sequence_length # size of a sequence to homogenize them
)

# Printing details about the variable
if not FLAG_SILENT_TRUE:
    print("'vectorize_layer'")
    personal_functions.print_variable_information(vectorize_layer, "vectorize_layer")
    print() # Esthetic

# Fill the layer with vocabulary from the dataset (text_ds)
# batch(batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None) -> 'DatasetV2' : Combines consecutive elements of this dataset into batches.
vectorize_layer.adapt(text_ds.batch(batch_size))

# Printing details about the variable
if not FLAG_SILENT_TRUE:
    print("'vectorize_layer' : After filling it with vocabulary")
    personal_functions.print_variable_information(vectorize_layer, "vectorize_layer")
    print() # Esthetic

# Save the created vocabulary for reference.
# TextVectorization.get_vocabulary(include_special_tokens=True) : Returns the current vocabulary tokens of the layer, sorted (descending) by their frequency = First vocabulary token is the most seen.
inverse_vocab = vectorize_layer.get_vocabulary()
if not FLAG_SILENT_TRUE:
    # Reminder : '[UNK]' = The unknown_token is used when what remains of the token is not in the vocabulary, or if the token is too long.
    NUMBER_OF_VOCABULARY_WORDS_TO_PRINT = 20 
    print("First " + str(NUMBER_OF_VOCABULARY_WORDS_TO_PRINT) + " words in the vocabulary")
    print(inverse_vocab[:NUMBER_OF_VOCABULARY_WORDS_TO_PRINT])
    print() # Esthetic

# Vectorize the data (generate one vector) in 'text_ds' by mapping it with the 'vectorize_layer'
# Reminder : text_ds = tf.data.Dataset
# From 'text_ds' -> Process by batch of batch_size
# Then, prefetch (pre-analyse, pre-read) with AUTOTUNE (see below)
# AUTOTUNE : If the value tf.data.AUTOTUNE is used, then the number of parallel calls is set dynamically based on available resources.
# Then, map those words with the vectorize_layer
# And unbatch : re-merge them. 
text_vector_ds = text_ds.batch(batch_size).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

# Generate a list of sequences (= vectors of same size) from the 'text_vector_ds'
# as_numpy_iterator() : Returns an iterator which converts all elements of the dataset to numpy.
sequences = list(text_vector_ds.as_numpy_iterator())

# Showing the lenght of sequences as well as some of them
if not FLAG_SILENT_TRUE:
    print("The lenght of the 'sequences' object is : " + str(len(sequences)))
    NUMBER_OF_SEQUENCES_TO_PRINT = 3
    print("Example of the first " + str(NUMBER_OF_SEQUENCES_TO_PRINT) + " sequences content :")
    for i in range(0, NUMBER_OF_SEQUENCES_TO_PRINT): 
        print(f"Sequence {i} : {sequences[i]} equivalent to : {[inverse_vocab[j] for j in sequences[i]]}\n")
    print() # Esthetic




# I AM HERE





"""
# My test data while developing
sequences = [[3, 5], [1, 7], [5, 6], [4, 5], [1, 6], [4, 2], [7, 1], [2, 1], [4, 3], [5, 3], [1, 3], [7, 6], [5, 4], [6, 1], [5, 1], [3, 2], [4, 1], [1, 5], [3, 1], [1, 2], [2, 4], [1, 4], [3, 4], [2, 3], [6, 5], [6, 7]]
window_size = 2
num_ns = 4 # number of negative samples
vocab_size = 8
seed = 42 # to adapt

#generate_training_data(sequences, window_size, num_ns, vocab_size, seed)
"""

##############################################

# My remarks to be processed before leaving the script
print("\n\n\nDEBUG AND REMARKS TO MYSELF WHILE DEVELOPPING")
print("##################################################")
print("Continue from the section 'Generate training examples from sequences'")
print("Re-check the function above ' generate_training_data' after writing the whole code to understand it.")
print("##################################################")
print("TO GO FURTHER")
print("\t- If needed : see other detailed explanation of skip gram model here :" + "\n" + "\thttps://medium.com/@corymaklin/word2vec-skip-gram-904775613b4c")
print("\t- See also : tutorial on how to use word2vec in gensim :" + "\n" + "\thttps://rare-technologies.com/word2vec-tutorial/")
print("##################################################")
print() # Esthetic
