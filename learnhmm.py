import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)

    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}

    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}

    return train_data, words_to_indices, tags_to_indices, args.hmminit, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()

    # Initialize the initial, emission, and transition matrices
    num_tags = len(tags_to_index)
    num_words = len(words_to_index)
    init_matrix = np.zeros(num_tags)
    emit_matrix = np.zeros((num_tags, num_words))
    trans_matrix = np.zeros((num_tags, num_tags))

    # Increment the matrices
    for example in train_data:
        prev_tag = None
        for word, tag in example:
            word_index = words_to_index[word]
            tag_index = tags_to_index[tag]
            emit_matrix[tag_index, word_index] += 1
            if prev_tag is not None:
                trans_matrix[prev_tag, tag_index] += 1
            prev_tag = tag_index
        init_matrix[tags_to_index[example[0][1]]] += 1

    # Add a pseudocount
    init_matrix += 1
    emit_matrix += 1
    trans_matrix += 1

    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    np.savetxt(init_out, init_matrix / np.sum(init_matrix), delimiter=" ")
    np.savetxt(emit_out, emit_matrix / np.sum(emit_matrix, axis=1)[:, None], delimiter=" ")
    np.savetxt(trans_out, trans_matrix / np.sum(trans_matrix, axis=1)[:, None], delimiter=" ")

    pass
