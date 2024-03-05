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

        # I AM HERE
        
        # Build context and label vectors (for one target word)
        context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_ns, dtype="int64")
      
        # Append each element from the training example to global lists.
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

    return targets, contexts, labels
    
######### DESCRIPTION OF FILE ################

if not FLAG_SILENT_TRUE:
    print("\nThis code is inspired from the website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb \nIt shows how to build a 'Continuous skip-gram model' which is able to predict words within a certain range before and after the current word in the same sentence. \nWhile a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context of target_word.\n")
    
############## MAIN CODE  ####################

# My test data while developing
sequences = [[3, 5], [1, 7], [5, 6], [4, 5], [1, 6], [4, 2], [7, 1], [2, 1], [4, 3], [5, 3], [1, 3], [7, 6], [5, 4], [6, 1], [5, 1], [3, 2], [4, 1], [1, 5], [3, 1], [1, 2], [2, 4], [1, 4], [3, 4], [2, 3], [6, 5], [6, 7]]
window_size = 2
num_ns = 4 # number of negative samples
vocab_size = 100
seed = 42 # to adapt

generate_training_data(sequences, window_size, num_ns, vocab_size, seed)





##############################################

# My remarks to be processed before leaving the script
print("\n\n\nDEBUG AND REMARKS TO MYSELF WHILE DEVELOPPING")
print("##################################################")
print("Finish to study/comment the function on line 111 : '# I AM HERE'")
print("Then follow with the section 'Prepare training data for word2vec'")
print("##################################################")
print("TO GO FURTHER")
print("\t- If needed : see other detailed explanation of skip gram model here :" + "\n" + "\thttps://medium.com/@corymaklin/word2vec-skip-gram-904775613b4c")
print("\t- See also : tutorial on how to use word2vec in gensim :" + "\n" + "\thttps://rare-technologies.com/word2vec-tutorial/")
print("##################################################")
print() # Esthetic
