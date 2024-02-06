import gzip
import os
import pathlib
import shutil
import sys
import urllib
import zipfile

# This file contain my personal functions
# They are general in order to be reused
# in different contexts.

def create_dir_if_not_existing(dir_path):
    # Try to create dir (if not existing) and return the path or exit with error
    try:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            print("\nDir \"" + dir_path + "\" created !\n")
        else:
            print("\nThe dir \"" + dir_path + "\" was already existing.\n")
        return dir_path
    except:
        print("\nERROR : the dir \"" + dir_path + "\" could not be created. Exiting...\n")
        sys.exit(1)

def download_data_file_from_url(data_file_url, path_data_filename):
    # Download the file and return a tuple :
    # (path to your output file, HTTP message object)
    urllib.request.urlretrieve(data_file_url, path_data_filename)
    return None
        
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
        print("Key : " + str(key) + " / Value : " + str(value))
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

def unzip_data_file(data_file_path, binary_true=False):
    """
    Returns the dir where the file was unzipped or exit with error if did not recognize the extension.
    """  
    # Adaptation of write and read mode if binary file
    READ_CUSTOM_MODE = "r"
    WRITE_CUSTOM_MODE = "w"
    if binary_true==True:
        READ_CUSTOM_MODE += "b"
        WRITE_CUSTOM_MODE += "b"

    # Check if the data file path provided exists - exit with error else
    if not os.path.exists(data_file_path):
        print("\nERROR : the path \"" + data_file_path + "\" does not exist. \nPlease check it and re-try after.\nExiting...\n")
        sys.exit(1)
    
    # Get the file extension from the path
    file_extension = str(pathlib.Path(data_file_path).suffix)

    # Check if the extension is known - add one for each condition below
    EXTENSION_LIST = [ ".gz", ".gzip", ".zip" ]
    FLAG_EXTENSION_IN_EXTENSION_LIST_TRUE = False
    for EXTENSION_NAME in EXTENSION_LIST:
        if file_extension == EXTENSION_NAME:
            FLAG_EXTENSION_IN_EXTENSION_LIST_TRUE = True
            break
    # If extension not know, print error message and exit
    if not FLAG_EXTENSION_IN_EXTENSION_LIST_TRUE:    
        print("\nERROR : the filename extension \"" + file_extension + "\" is unknown. Check the filename or develop a condition for processing such filename extension...\n")
        sys.exit(1)

    # Get the target name of unzipped dir = filename without extension
    unzipped_target_dir = str(pathlib.Path(data_file_path).with_suffix(''))
    # Create it if not already existing
    create_dir_if_not_existing(unzipped_target_dir)
    # Get the simulated file basename
    unzipped_file_basename = os.path.basename(unzipped_target_dir)
    # Get the simulated unzipped filename = target dir + basename
    unzipped_file_path = unzipped_target_dir + "/" + unzipped_file_basename
    
    # gzip
    if ( file_extension == ".gz" or file_extension == ".gzip" ):
        # Check if an unzipped file with same name does not already exists
        if not os.path.isfile(unzipped_file_path):
        # If not already exists, unzip it
            with gzip.open(data_file_path, READ_CUSTOM_MODE) as file_in:
                with open(unzipped_file_path, WRITE_CUSTOM_MODE) as file_out:
                    print("\nUnzipping the binary file located on path \"" + data_file_path + "\"")
                    shutil.copyfileobj(file_in, file_out)
                    print("Binary file \"" + unzipped_file_basename + "\" located \"" + unzipped_file_path + "\" obtained after successfully unzipping !\n")
        else:
            print("\nAn unzipped file with the name \"" + unzipped_file_basename + "\" located \"" + unzipped_file_path + "\" already exists.\nIn case it is not wanted, please remove/rename it and re-try after...\n")

    # zip
    if file_extension == ".zip":
        with zipfile.ZipFile(data_file_path, READ_CUSTOM_MODE) as zip_ref:
            # Unzip in target dir
            zip_ref.extractall(unzipped_target_dir)
            
    # Returning the dir where files were unzipped to use it outside the function      
    return unzipped_target_dir
