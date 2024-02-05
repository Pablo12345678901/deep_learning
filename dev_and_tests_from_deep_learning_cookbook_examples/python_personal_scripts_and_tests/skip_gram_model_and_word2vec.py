import io # Core tools for working with streams
import re # Regex operations
import string
import tqdm # Shows progress bar in Python

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# My custom imports
import personal_functions

#####################################################

"""
This code is inspired from the website : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb 

It shows how to build a 'Continuous skip-gram model' which is able to predict words within a certain range before and after the current word in the same sentence.

While a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context of target_word.
"""

# Setup
SEED = 42
# tf.data : an API that enables to build complex input pipelines from simple, reusable pieces. 
AUTOTUNE = tf.data.AUTOTUNE # = -1

# Vectorize an example sentence
sentence = "The wide road shimmered in the hot sun"
# Below sentence : shows that token can contains the char "'" and does not split 'I'm' into token[0]='I' and token[1]='m'  
#sentence = "I'm the king of my wife's dog."
# Split (with the space char) the sentence into a 'tokens' list.
# And lower all chars - no difference between 'The', 'the' - both consideredas 'the'
tokens = list(sentence.lower().split())
# Show the lenght of the 'tokens' list and its content, index per index
#personal_functions.printing_list_elements(tokens)



# My remarks to be processed before leaving the script
print() # Esthetic
print("DEBUG : I AM HERE")
print("I'm in the 'Vectorize an example sentence' part")
print("What is 'SEED' and why set it to '42' ?")
print("What is 'AUTOTUNE' and why set it to '-1' ?")
input("DEBUG : do the above before finishing this script") 
