# This file contain my personal functions
# They are general in order to be reused
# in different contexts.

def printing_list_elements(list):
    list_lenght = len(list)
    print("\nPrinting " + str(list_lenght) + " elements of the list :")
    for i in range(list_lenght):
        print("Index : " + str(i) + " / Value : " + str(list[i]))
    print() # Esthetic

# Below : customs list of dict functions to print them in a pretty way
def print_dictionary_elements(dictionary, index=0):
    print("Index of dictionary : " + str(index))
    for key, value in dictionary.items():
        print("Key : " + key + " / Value : " + str(value))
    print() # Esthetic

def print_list_of_dictionaries(list_of_dictionaries):
    for i in range(len(list_of_dictionaries)):
        print_dictionary_elements(list_of_dictionaries[i], i)

def print_sorted_list_of_dictionaries_by_a_key_value(list_of_dictionaries, value):
    newlist = sorted(list_of_dictionaries, key=itemgetter(value), reverse=False)
    print("\nList of dictionaries sorted by the value of the key " + value)
    for i in range(len(newlist)):
        print_dictionary_elements(newlist[i], i)

def printing_tuple(tuple, index=0):
    string_to_print = str(tuple[0])
    if len(tuple)>1:
        for i in range(1, len(tuple)):
            string_to_print = string_to_print + ", " + str(tuple[i]) # in case there is number
    string_to_print = "Index : " + str(index) + " / Value of tuple : ( " + string_to_print + " ) "
    #print(str(index) + " - " + string_to_print)
    print(string_to_print)
    return string_to_print
    #print() # Esthetic

def printing_list_of_tuples(list_of_tuples):
    for i in range(len(list_of_tuples)):
        printing_tuple(list_of_tuples[i], i)
    print() # Esthetic

def printing_zip_elements(zip_object):
    # Creating a copy of initial zip object
    copy_of_zip_object = zip_object
    # Converts to list in order to show
    copy_of_zip_object = list(copy_of_zip_object)
    for i in range(len(copy_of_zip_object)):
        print("Index : " + str(i) + " / Value of zip element : " + str(copy_of_zip_object[i]))
    print() # Esthetic
