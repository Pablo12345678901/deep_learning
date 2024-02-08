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

#####################################################

"""
This code is inspired from the website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb 

It shows how to build a 'Continuous skip-gram model' which is able to predict words within a certain range before and after the current word in the same sentence.

While a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context of target_word.
"""

##############################################

# By default, print additional output messages to help the script understanding
# Choose to print or not output messages + values at several steps
FLAG_SILENT_TRUE = False

# Manage options provided to the script
MAXIMUM_ARGUMENT_NUMBER = 1
# Check if one argument was provided
NUMBER_OF_PROVIDED_ARGUMENTS = len(sys.argv) - 1 # -1 because arg[0] = script name
# Reminder :
#    sys.argv[0] = name of the script
#    sys.argv[1] = first argument ... and so on
if NUMBER_OF_PROVIDED_ARGUMENTS == 1:
    ARGUMENT_PROVIDED = str(sys.argv[1])
    if ARGUMENT_PROVIDED == "--silent": 
        FLAG_SILENT_TRUE = True
    elif ARGUMENT_PROVIDED == "--verbose":
        FLAG_SILENT_TRUE = False
    else:
        print("\nERROR : unknown argument provided : " + ARGUMENT_PROVIDED + ".\n")
        sys.exit(1)
elif NUMBER_OF_PROVIDED_ARGUMENTS > MAXIMUM_ARGUMENT_NUMBER:
    print("\nERROR : too many arguments provided : " + str(sys.argv[1:]) + ".\n")
    sys.exit(2)

##############################################

# Setup
# Set the random seed
SEED = 42
# tf.data : an API that enables to build complex input pipelines from simple, reusable pieces. 
AUTOTUNE = tf.data.AUTOTUNE # = -1

##############################################

# Vectorize an example sentence
sentence = "The wide road shimmered in the hot sun"
# Below sentence : shows that token can contains the char "'" and does not split 'I'm' into token[0]='I' and token[1]='m'  
#sentence = "I'm the king of my wife's dog."
# Split (with the space char) the sentence into a 'tokens' list.
# And lower all chars - no difference between 'The', 'the' - both consideredas 'the'
tokens = list(sentence.lower().split())
# Show the lenght of the 'tokens' list and its content, index per index
if not FLAG_SILENT_TRUE:
    print("\nValue of the 'tokens' list")
    personal_functions.printing_list_elements(tokens)

# Define a 'vocab' dict and index (int)
vocab, index = {}, 0
# Add a padding token with value 0 because the token of index 0 will be skipped by the tf.keras.preprocessing.sequence.skipgrams function.
vocab['<pad>'] = index
# Increase the index each time a key is added into the dict
# That way, all (unique) keys have a unique index as as unique identifier
index += 1
# Add all tokens into the dict
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1

# Get the number of keys in the dict
vocab_size = len(vocab)
# Below : shows the dict = its content and its lenght
if not FLAG_SILENT_TRUE:
    print("Dict 'vocab' : each token (=key) has an unique index (=value)")
    print("The dict of tokens has a lenght of : " + str(vocab_size) + "\n")
    personal_functions.print_dictionary_elements(vocab)

# Create an inverse dict with indexes as keys and tokens as keys values
inverse_vocab = {index: token for token, index in vocab.items()}
inverse_vocab_size = len(inverse_vocab)
# Below : shows the dict = its content and its lenght
if not FLAG_SILENT_TRUE:
    print("Dict 'inverse_vocab' : each index (=key) has an unique token (=value)")
    print("The dict of tokens has a lenght of : " + str(inverse_vocab_size) + "\n")
    personal_functions.print_dictionary_elements(inverse_vocab)

# Vectorize the sentence by replacing all token use with their indexes
example_sequence = [vocab[word] for word in tokens]
# Show the vector created
if not FLAG_SILENT_TRUE:
    print("Printing the equivalence of the sentence : '" + sentence + "' when vectorized by replacing each token by its key value.", "\n")
    print(example_sequence, "\n")
    # Show the equivalent tokens to the vector from the original sentence
    equivalent_tokens = [ word for word in tokens ]
    print(equivalent_tokens, "\n")

##############################################

# Generate skip-grams from one sentence

window_size = 2
# tf.keras.preprocessing.sequence module provides useful functions that simplify data preparation for word2vec.
# tf.keras.preprocessing.sequence.skipgrams : Generates skipgram word pairs.
"""
Function definition : 
tf.keras.preprocessing.sequence.skipgrams(
    sequence,
    vocabulary_size,
    window_size=4,
    negative_samples=1.0,
    shuffle=True,
    categorical=False,
    sampling_table=None,
    seed=None
)
"""
# 'vocabulary_size' : Int, maximum possible word index + 1. Note that index 0 is expected to be a non-word and will be skipped.
# 'negative_samples' is set to 0 as batching negative samples generated by this function requires a bit of code. Another function will be used to perform negative sampling in the next section.
# 'window_size' : number of token considered on each size of the token processed (so the number of token considered is :
# = windows_size + token_processed + windows_size
# = 2*windows_size + 1
# Here : the function returns :
#    couples, labels : where couples are int pairs and labels are either 0 or 1 - 0 for negative samples.
# At this step, we ignore the labels part stocked into '_' because there are no negative sample so all labels equals '1'.
window_size = 2
positive_skip_grams, labels_for_skip_grams = tf.keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)
# Show the results returned by the function tf.keras.preprocessing.sequence.skipgrams
if not FLAG_SILENT_TRUE:
    print("Those are the skip grams (=couples of int) generated from the sentence with a windows of " + str(window_size) + " words around the context word :" + "\n" + repr(positive_skip_grams) + "\n") # return a list of couples with indexes of tokens
# Ex :[[3, 2], [2, 3], [4, 5], [1, 7], [5, 6], [3, 5], [5, 1], [3, 1], [1, 3], [1, 2], [1, 5], [4, 1], [1, 6], [7, 1], [6, 1], [5, 4], [6, 5], [3, 4], [6, 7], [2, 1], [1, 4], [7, 6], [5, 3], [4, 3], [4, 2], [2, 4]]
    # Printing what is the equivalent of each int couple translated in tokens.
    print("And those couples of int equivault to : ", "\nint couple", "/", "skip-gram")
    for target, context in positive_skip_grams[:]:
        print(f"[{target}, {context}] : [{inverse_vocab[target]}, {inverse_vocab[context]}]")
    print() # Esthetic
    print("The 'labels_for_skip_grams' list for each int couples can be ignored here because at this step, we did not use negative samples.\n" + repr(labels_for_skip_grams) + "\n") # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print("In total there are (lenght) :", len(positive_skip_grams), "skip grams in the 'positive_skip_grams' list.", "\n") # = 26

##############################################

# Negative sampling for one skip-gram

# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context.
num_of_negative_samples = 4

"""
tf.constant : Creates a constant tensor from a tensor-like object.

tf.constant(
    value, dtype=None, shape=None, name='Const'
) -> Union[tf.Operation, ops._EagerTensorBase]
"""
CONSTANT_TENSOR = tf.constant(context_word, dtype="int64")
if not FLAG_SILENT_TRUE:
    print("Constant tensor : " + "\n" + str(CONSTANT_TENSOR) + " -> " + str(type(CONSTANT_TENSOR)) + "\n" + "created from tensor object : " + "\n" + str(context_word) + " -> " + str(type(context_word)) + "\n")
    
"""
tf.reshape : Reshapes a tensor.

tf.reshape(
    tensor, shape, name=None
)
"""
context_class = tf.reshape(CONSTANT_TENSOR, (1, 1))
if not FLAG_SILENT_TRUE:
    print("Context class generated from context word (=token index): " + "\n" + str(context_class) + " -> " + str(type(context_class)) + "\n")
    
"""
tf.random.log_uniform_candidate_sampler returns 3 elements :
    sampled_candidates : a set of classes ('tf.Tensor' class objects) using a log-uniform (Zipfian) base distribution.
    true_expected_count : represents the number of times each of the target classes (true_classes) occurs in an average tensor of sampled classes. Here, true_classes = context_class = only one 'positive' class.
    sampled_expected_count : represents the number of times each of the sampled classes (sampled_candidates) were expected to occur in an average tensor of sampled classes.

So it returns candidates samplers and also the frequency they really appeared in average as well as the frequency they were expected to appear. Those two later are ignored here.

Tf.tensor represents a multidimensional array of elements.

Further on Zipfian law : https://en.wikipedia.org/wiki/Zipf's_law
The base distribution for this operation is an approximately log-uniform or Zipfian distribution:
P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
"""
negative_sampling_candidates, true_expected_count_list, sampled_expected_count_list = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  # class that should be sampled as 'positive'
    num_true=1,  # each positive skip-gram has 1 positive context class
    num_sampled=num_of_negative_samples,  # number of negative context words to sample
    unique=True,  # all the negative samples should be unique - when one is taken, it is withdrawn from the list of samples that could be choosed next.
    range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
    seed=SEED,  # seed for reproducibility - if you repeat the operation with the same seed, you will get the same 'negative_sampling_candidates', whatever is the 'true_classes'
    name="negative_sampling"  # name of this operation
)

# Showing in details the negatives sampling candidates, their key value (int) and their equivalent if tokens
if not FLAG_SILENT_TRUE:
    print("Get negative sampling candidates from the context class list composed of index objects:")
    personal_functions.printing_list_elements(negative_sampling_candidates)
    print("Showing the index objects of the 'negative_sampling_candidates' list and their int value obtained after being converted to numpy objects")
    list_of_indexes = []
    # Actually, here, the 'tf.Tensor' object is actually a 'EagerTensor' one.
    # So we have access to the 'numpy()' function which concerts it to a Numpy array (or one element if there is only one).
    for index in negative_sampling_candidates:
        print("index object : \n" + str(index) + " -> " + str(type(index)) + "\n" + "converted to a Numpy array (of one element) with index.numpy() : \n" + repr(index.numpy()) + " -> object type : " + str(type(index.numpy())) )
        list_of_indexes.append(index.numpy())
    print() # Esthetic
    print("Showing what are the tokens equivalent to ", list_of_indexes)
    print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates], "\n")
    print("The positive class 'context_class' :" + "\n" + str(context_class) + "\n" + "represented by the int : " + str(context_word) + " -> " + str(type(context_word)) + "\n" + "was expected to appear in average :" + "\n" + str(true_expected_count_list) + " -> " + str(type(true_expected_count_list)) + "\n") # See below
    # For each of the 7 tokens from the sentence above, the results are :
    # 1 = [[0.6393995]
    # 2 = [0.5042367]
    # 3 = [0.41460013]
    # 4 = [0.35151714]
    # 5 = [0.30489868]
    # 6 = [0.26910767]
    # 7 = [0.24079101]
    print("The samples candidates selected " + str(list_of_indexes) + " were expected to appear in average :" + "\n" + str(sampled_expected_count_list) + " -> " + str(type(sampled_expected_count_list)) + "\n")
    
##############################################

# Construct one training example
























##############################################

# My remarks to be processed before leaving the script
print("\n\n\nDEBUG AND REMARKS TO MYSELF WHILE DEVELOPPING")
print("\t- What is 'AUTOTUNE' (line 57) and why set it to '-1' ?")
print("##################################################")
print("OK")
print("##################################################")
print("TO GO FURTHER")
print("\t- If needed : see other detailed explanation of skip gram model here :" + "\n" + "\thttps://medium.com/@corymaklin/word2vec-skip-gram-904775613b4c")
print("\t- See also : tutorial on how to use word2vec in gensim :" + "\n" + "\thttps://rare-technologies.com/word2vec-tutorial/")
print("##################################################")
print() # Esthetic
