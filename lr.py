import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def dJ(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int
) -> np.ndarray: 
    
    # Calculate the gradient at the i-th sample
    x = X[i, :]
    grad = (sigmoid(np.dot(theta, x)) - y[i]) * x
    
    return grad


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> np.ndarray:
    
    # SGD 
    for epoch in range(num_epoch): 
        indices = range(X.shape[0])
        for i in indices:
            theta -= learning_rate * dJ(theta, X, y, i)
    
    return theta
    
    pass


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    
    predict_list = []
    
    for i in range(X.shape[0]):
        x = X[i, :]
        res = np.dot(theta, x)
        prediction = sigmoid(res)
        if prediction >= 0.5:
            y = 1
        else: 
            y = 0
        predict_list.append(y)
        
    predict_array = np.array(predict_list)
    
    return predict_array

    pass


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    
    err_array = np.abs(y_pred - y)
    err = err_array.sum()
    err_rate = err / (err_array.shape[0])
    
    return err_rate
    
    pass


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str, 
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str, 
                        help='learning rate for gradient descent')
    args = parser.parse_args()
    
    train_input = np.genfromtxt(args.train_input, delimiter = '\t', dtype = None, encoding = None)
    validation_input = np.genfromtxt(args.validation_input, delimiter = '\t', dtype = None, encoding = None)
    test_input = np.genfromtxt(args.test_input, delimiter = '\t', dtype = None, encoding = None)
    
    X_train = train_input[:, 1:]
    X_train_withones = np.insert(X_train, 0, 1, axis = 1)
    y_train = train_input[:, 0]
    
    X_test = test_input[:, 1:]
    X_test_withones = np.insert(X_test, 0, 1, axis = 1)
    y_test = test_input[:, 0]
    
    train_out = args.train_out 
    test_out = args.test_out 
    metrics_out = args.metrics_out
    num_epoch = int(args.num_epoch)
    lr = float(args.learning_rate)
    theta = np.zeros(VECTOR_LEN + 1)
    
    trained_theta = train(theta, X_train_withones, y_train, num_epoch, lr)
    predict_train = predict(trained_theta, X_train_withones)
    predict_test = predict(trained_theta, X_test_withones)
    
    train_err_rate = compute_error(predict_train, y_train)
    test_err_rate = compute_error(predict_test, y_test)
    
    # Write outputs in txt files 
    with open(train_out, 'w') as f:
        for i in predict_train:
            f.write(str(i) + '\n')
            
    with open(test_out, 'w') as f:
        for i in predict_test:
            f.write(str(i) + '\n')
            
    with open(metrics_out, 'w') as f:
        f.write('error(train): ' + "%.6f" % train_err_rate + '\n')
        f.write('error(test): ' + "%.6f" % test_err_rate)
    
    
