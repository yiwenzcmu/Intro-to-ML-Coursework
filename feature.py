import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def convert(dataset, wordmap):
    """
    
    Parameters:
        dataset: An np.ndarray of shape N storing orginal dataset.
        wordmap: A dictionary for mapping each word with a feature vector. 

    Returns:
        A list of tuples, one tuple for label and ndarray of feature of each sample.

    """
    label_list = [] 
    feature_list = [] 
    data_size = dataset.shape[0]
    for i in range(data_size):     
        sample = dataset[i]
        label = sample[0]
        label_list.append(label)
        sen = sample[1]
        word_arr = np.array(sen.split())
        sen_trim = np.array([word for word in word_arr if word in wordmap])
        total = np.zeros(VECTOR_LEN)
        for j in sen_trim:
            total += wordmap[j]
                        
        feature = total / sen_trim.shape[0]
        feature_list.append(feature)
    
    output = list(zip(label_list, feature_list))  
    return output   

def write_tsv(output, outputfile):
    
    """
    

    Parameters
    ----------
    output : A list of tuples, one tuple for label and ndarray of feature of each sample.
    outputfile : String
        DESCRIPTION.

    Returns
    -------
    None.

    """ 
    with open(outputfile, 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in output:
            # Join the elements in the numpy array with tabs
            array_str = '\t'.join(np.char.mod('%.6f', row[1]))
            # Convert the tab-separated string back to a list
            array_list = array_str.split('\t')
            # Write the row to the TSV file
            writer.writerow(('%.6f'% row[0], *array_list))                                  



if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    
    train_data = load_tsv_dataset(args.train_input)
    validation_data = load_tsv_dataset(args.validation_input)
    test_data = load_tsv_dataset(args.test_input)
    
    glove_map = load_feature_dictionary(args.feature_dictionary_in)
    
    train_output = convert(train_data, glove_map)
    validation_output = convert(validation_data, glove_map)
    test_output = convert(test_data, glove_map)
    
    formatted_train = args.train_out
    formatted_valid = args.validation_out
    formatted_test = args.test_out
    
    write_tsv(train_output, formatted_train)
    write_tsv(validation_output, formatted_valid)
    write_tsv(test_output, formatted_test)
    
    
