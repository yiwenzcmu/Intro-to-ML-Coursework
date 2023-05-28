# -*- coding: utf-8 -*-

import sys
import numpy as np

# Define Node Class
class Node():
    def __init__(self, attr_index = None, left = None, right = None, info_gain = None, vote = None):
        # for internal node
        self.attr_index = attr_index
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf node
        self.vote = vote
        
    def is_leaf_node(self):
        if self.vote is not None:
            return True
        else:
            return False
        
# Define Decision Tree Classifier Class
class DecisionTreeClassifier():
    '''
    Class that implements a decsion tree classifier algorithm
    '''
    def __init__(self, max_depth = 100):
        self.root = None
        self.max_depth = max_depth
    
    def _is_pure(self, y):
        if len(np.unique(y)) == 1:
            return True
        
    def _split(self, X_attr):
        left_idxs = np.where(X_attr == 1)[0]
        right_idxs = np.where(X_attr == 0)[0]
        return left_idxs, right_idxs    
    

    def _majority_vote(self, y):
        num_ones = np.count_nonzero(y == 1)
        num_zeros = len(y) - num_ones
        if num_zeros > num_ones:
            vote = 0
        else:
            vote = 1        
        return vote
    
    
    def _entropy(self, y):
        num_ones = np.count_nonzero(y == 1)
        num_zeros = len(y) - num_ones
 
        if num_zeros == 0 or num_ones == 0:
            entropy = 0
        else: 
            ones_pct = float(num_ones / len(y))
            zeros_pct = float(num_zeros / len(y))
            entropy = - ones_pct * np.log2(ones_pct) - zeros_pct * np.log2(zeros_pct)
        return entropy
 
    
    def _information_gain(self, y, X_attr): 
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_attr)
                                       
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        left_prop = float(len(left_idxs) / len(y))
        right_prop = float(len(right_idxs) / len(y))
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        child_entropy = left_entropy * left_prop + right_entropy * right_prop
        
        # calculate the mutual information
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    
    def _best_split(self, X, y, available_attr):
        best_gain = 0
        best_attr_idx = 0        
        for attr_idx in range(len(available_attr)):
            if available_attr[attr_idx] == 1:
                X_attr = X[:, attr_idx]
                information_gain = self._information_gain(y, X_attr)            
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_attr_idx = attr_idx   
        return best_attr_idx, best_gain
    
    
    def _build_tree(self, X, y, available_attr, curr_depth = 0):

        if (self._is_pure(y) or max_depth ==0 or curr_depth > max_depth -1 or curr_depth > len(available_attr) - 1):
            leaf_value = self._majority_vote(y)
            
            str_print = ""
            for i in range(curr_depth):
               str_print += "| "
            # print []
            str_print += "[" + str(sum(y==0)) + " 0/" + str(sum(y==1)) + " 1]"
            print(str_print)
          
            return Node(vote = leaf_value)
        else: 
            # find the best split 
            best_attr_idx, best_gain = self._best_split(X, y, available_attr)
            if best_gain > 0:  
                left_idxs, right_idxs = self._split(X[:, best_attr_idx])
                available_attr[best_attr_idx] = 0
               
                # start to print
                # print indent
                str_print = ""
                for i in range(curr_depth):
                    str_print += "| "
                # print attribute name
                str_print += ( str(best_attr_idx) + "-th attr : " )
                # print []
                str_print += "[" + str(sum(y==0)) + " 0/" + str(sum(y==1)) + " 1]"
                print(str_print)
                
                left = self._build_tree(X[left_idxs, :], y[left_idxs], available_attr.copy(), curr_depth + 1)                
                right = self._build_tree(X[right_idxs, :], y[right_idxs], available_attr.copy(), curr_depth + 1)
                return Node(best_attr_idx, left, right, best_gain)
            else:
                leaf_value = self._majority_vote(y)
                return Node(vote = leaf_value)
 
    ''' Training '''
    def fit(self, X, y):
        n_attributes = X.shape[1]
        available_attr = [1] * n_attributes 
        self.root = self._build_tree(X, y, available_attr)
    
    ''' Print the tree '''
    #def print_tree():
        #print ("[" + "{}")
        
    ''' Prediction '''
    def predict(self, X):
       # loop over every sample's atttributes
       predictions = []
       for i in X:
           predictions.append(self.make_predictions(i, self.root))
       return predictions
   
    def make_predictions(self, x, node):
        if node.is_leaf_node():
            return node.vote
        elif x[node.attr_index] == 1:
            return self.make_predictions(x, node.left)
        else:
            return self.make_predictions(x, node.right)
    
    ''' Evaluation'''
    def calc_error (self, predictions, true):
        err_cnt = 0
        for i in range(len(predictions)):
            if predictions[i] != int(true[i]):
                err_cnt += 1
        error_rate = err_cnt / len(predictions)
        return error_rate
    
 



    

if __name__ == "__main__":
    args = sys.argv    
    # Parse every argument
    train_input = args[1]
    test_input = args[2]
    max_depth = int(args[3])
    train_out = args[4]
    test_out = args[5]
    metrics_out = args[6]
    
    # Prepare data
    train_data = np.genfromtxt(train_input, delimiter = '\t', dtype = None, encoding = None)
    test_data = np.genfromtxt(test_input, delimiter = '\t', dtype = None, encoding = None)
    X_train = train_data[1:][:,:-1].astype('int')
    X_test = test_data[1:][:,:-1].astype('int')
    y_train = train_data[1:][:, -1].astype('int')
    y_test = test_data[1:][:, -1].astype('int')
    attributes = train_data[0, :-1] 
    
    # consider the case that specified max_depth larger than number of attributes
    if max_depth > len(attributes):
        max_depth = len(attributes)
    
    clf = DecisionTreeClassifier(max_depth)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    train_error_rate = clf.calc_error(train_predictions, y_train)
    test_error_rate = clf.calc_error(test_predictions, y_test)
    # Write outputs in txt files 

    with open(train_out, 'w') as f:
        for i in train_predictions:
            f.write(str(i) + '\n')
    with open(test_out, 'w') as f:
        for i in test_predictions:
            f.write(str(i) + '\n')
    with open(metrics_out, 'w') as f:
        f.write('error(train): ' + "%.6f" % train_error_rate + '\n')
        f.write('error(test): ' + "%.6f" % test_error_rate)


    
    