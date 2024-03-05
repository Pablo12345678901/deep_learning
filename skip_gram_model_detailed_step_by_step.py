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

############# OPTIONS MANAGEMENT #######################

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

############## DESCRIPTION ####################

if not FLAG_SILENT_TRUE:
    print("\nThis code is inspired from the website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb \nIt shows how to build a 'Continuous skip-gram model' which is able to predict words within a certain range before and after the current word in the same sentence. \nWhile a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context of target_word.\n")

############# MAIN CODE #####################

# Setup

# Set the random seed
SEED = 42

# tf.data : an API that enables to build complex input pipelines from simple, reusable pieces.
# tf.data.AUTOTUNE : use tensorflow capacity to compute the processing time at each step while processing the input in order to optimize the pipeline while using this output (providing a better CPU usage) - more fluent.
# See : https://www.tensorflow.org/api_docs/python/tf/data#AUTOTUNE
# See : https://stackoverflow.com/questions/56613155/tensorflow-tf-data-autotune
AUTOTUNE = tf.data.AUTOTUNE # = -1

##############################################

# Vectorize an example sentence

sentence = "The wide road shimmered in the hot sun"
# Below sentence : shows that token can contains the char "'" and does not split 'I'm' into token[0]='I' and token[1]='m'  
#sentence = "I'm the king of my wife's dog." # Another sentence for tests.

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

# Add all tokens into the dict
for token in tokens:
  if token not in vocab:
    # Increase the index each time a key is added into the dict
    # That way, all (unique) keys have a unique index as identifier
    index += 1
    vocab[token] = index
    

# Get the number of keys in the dict
vocab_size = len(vocab)

# Shows the dict = its content and its lenght
if not FLAG_SILENT_TRUE:
    print("Dict 'vocab' : each token (=key) has an unique index (=value)")
    print("The dict of tokens has a lenght of : " + str(vocab_size) + "\n")
    personal_functions.print_dictionary_elements(vocab)

# Create an inverse dict with indexes as keys and tokens as keys values
inverse_vocab = {index: token for token, index in vocab.items()}
inverse_vocab_size = len(inverse_vocab)

# Shows the dict = its content and its lenght
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

# Set the number of words taken around the current one to create pairs.
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

# Show content of the 'constant tensor'
if not FLAG_SILENT_TRUE:
    print("Constant tensor : " + "\n" + str(CONSTANT_TENSOR) + " -> " + str(type(CONSTANT_TENSOR)) + "\n" + "created from tensor object : " + "\n" + str(context_word) + " -> " + str(type(context_word)) + "\n")
    
"""
tf.reshape : Reshapes a tensor.

tf.reshape(
    tensor, shape, name=None
)
"""
context_class = tf.reshape(CONSTANT_TENSOR, (1, 1))

# Show content of 'context class'
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

# Showing in details the negatives sampling candidates, their key value (int) and their equivalent in tokens
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
    print("The negative samples candidates selected " + str(list_of_indexes) + " were expected to appear in average :" + "\n" + str(sampled_expected_count_list) + " -> " + str(type(sampled_expected_count_list)) + "\n")
    
##############################################

# Construct one training example

# Reduce a dimension so you can use concatenation (in the next step).
"""
tf.squeeze : Removes dimensions of size 1 from the shape of a tensor.

tf.squeeze(
    input, axis=None, name=None
)
"""
squeezed_context_class = tf.squeeze(context_class, 1)

# Show the changes from 'context_class' to 'squeeted_context_class' 
if not FLAG_SILENT_TRUE:
    print("Created a new object class 'squeezed_context_class' from the 'context_class' one by reducing its dimensions to only one : ")
    print("Before : ", end='')
    personal_functions.print_variable_information(context_class, "context_class")
    print("After  : ", end='')
    personal_functions.print_variable_information(squeezed_context_class, "squeezed_context_class")
    print() # Esthetic

# Concatenate a positive context word with negative sampled words.
"""
tf.concat : Concatenates tensors along one dimension.

tf.concat(
    values, axis, name='concat'
)
"""
context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)
if not FLAG_SILENT_TRUE:
    print("Concatenate the two objects 'squeezed_context_class' and 'negative_sampling_candidates' into a 'context' object.\nThis will give a list of objects with the first one representing the positive word in the context of the target and the next ones the negatives : ")
    personal_functions.print_variable_information(context, "context")
    print() # Esthetic

# Label the first context word as `1` (positive) followed by `num_ns` `0`s (negative).
# tf.constant : see above -> Creates a constant tensor from a tensor-like object.
label = tf.constant([1] + [0]*num_of_negative_samples, dtype="int64")

# Shwo the content of the 'label object'
if not FLAG_SILENT_TRUE:
    print("Create a 'label' object containing a list of int - the first one is 1 (positive label) followed by 'num_of_negative_samples' " + str(num_of_negative_samples) + " times '0' (negative label) : ")
    personal_functions.print_variable_information(object=label, object_name="label")
    print() # Esthetic

target = target_word

# Summarize of important datas used above.
if not FLAG_SILENT_TRUE:
    print("Set the 'target' " + str(target) + " as per the 'target_word' " + str(target_word) + ".\nReminder : here, first word has index = 1.\n")
    # Some usefull info
    print(f"sentence        : {sentence}")
    print(f"target_index    : {target}")
    print(f"target_word     : {inverse_vocab[target_word]}")
    print(f"window size     : {window_size}")
    print(f"context_indices : {context}")
    print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
    print(f"label           : {label}")
    print(f"\nTo summarize, the context word '{inverse_vocab[context[0].numpy()]}' with token index {context_word} is in the context of the target word '{inverse_vocab[target_word]}' with token index {target} and the negative sampling candidates randomly taken (so should theorically but this is not certain) not to be in the context were {[inverse_vocab[c.numpy()] for c in context[1:]]}.\n")

"""
A tuple of (target, context, label) tensors constitutes one training example for training your skip-gram negative sampling word2vec model.

Notice that the target is of shape (1,) while the context and label are of shape (1+num_of_negative_samples,)
"""
if not FLAG_SILENT_TRUE:
    print("A tuple of (target, context, label) tensors constitutes one training example for training your skip-gram negative sampling word2vec model.\nNotice that the target is of shape (1,) while the context and label are of shape (1+num_ns,)")
    print("\nExample of a tuple of sensors :")
    print("target  :", target)
    print("context :", context)
    print("label   :", label)
    print() # Esthetic
    
##############################################

# End message as script can be long
print("\nEnd of script\n")
sys.exit(0)


