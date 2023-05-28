import argparse
import numpy as np
def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(x, axis = None):
    x_max = np.max(x, axis = axis, keepdims=True)
    return np.log(np.sum(np.exp(x-x_max), axis=axis, keepdims=True)) + x_max

def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    log_alpha = np.zeros((L,M))
    for t in range(L):
        for j in range(M):
            if t == 0:
                log_alpha[t][j] = loginit[j] + logemit[j][words_to_indices[seq[0]]]
            else:
                log_alpha[t][j] = logemit[j][words_to_indices[seq[t]]] + logsumexp(log_alpha[t-1, :] + logtrans[:,j])



    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    log_beta = np.zeros((L,M))
    for t in range(L-1, -1, -1):
        for j in range(M):
            if t == L-1:
                log_beta[L-1][j] = 1
            else:
                log_beta[t][j] = logsumexp(logemit[:, words_to_indices[seq[t+1]]] + log_beta[t+1][:] + logtrans[j,:])


    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the required tag
    log_posterior = log_alpha + log_beta
    predicted_tags = []
    for i in range(L):
        predicted_tags.append(list(tags_to_indices.keys())[np.argmax(log_posterior[i,:])])


    # Compute the stable log-probability of the sequence
    log_probability = logsumexp(log_alpha[-1])

    # Return the predicted tags and the log-probability
    return predicted_tags, log_probability
    pass
    

    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    print(validation_data)

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    total_log_likelihood = 0
    total_tags = 0
    correct_tags = 0

    loginit = np.log(hmminit)
    logemit = np.log(hmmemit)
    logtrans = np.log(hmmtrans)

    for example in validation_data:
        seq = [sublist[0] for sublist in example]
        predicted_tags, log_probability = forwardbackward(seq, loginit, logtrans, logemit, words_to_indices,
                                                          tags_to_indices)
        print(predicted_tags)
        total_log_likelihood += log_probability

        for i in range(len(seq)):
            total_tags += 1
            if predicted_tags[i] == example[i][1]:
                correct_tags += 1


    average_log_likelihood = total_log_likelihood / len(validation_data)
    accuracy = correct_tags / total_tags

    # Compute the average log-likelihood and the accuracy. The average log-likelihood
    # is just the average of the log-likelihood over all sequences. The accuracy is
    # the total number of correct tags across all sequences divided by the total number
    # of tags across all sequences.


    with open(predicted_file, 'w') as f:
        for example in validation_data:
            seq = [sublist[0] for sublist in example]
            predicted_tags, _ = forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices)
            for i in range(len(seq)):
                f.write(seq[i] + '\t' + predicted_tags[i] + '\n')
            f.write('\n')

    with open(metric_file, 'w') as f:
        f.write('Average Log-Likelihood: {}\n'.format(average_log_likelihood[0]))
        f.write('Accuracy: {}\n'.format(accuracy))


    pass