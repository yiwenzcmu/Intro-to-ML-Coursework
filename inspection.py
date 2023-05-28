# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:43:02 2023

@author: YiwenZhao0416
"""
import numpy as np
import sys

class Inspection():
    
    def __init__(self):
        return 
    
    def _entropy(self, y):
        num_ones = np.count_nonzero(y == '1')
        num_zeros = len(y) - num_ones
 
        if num_zeros == 0 or num_ones == 0:
            entropy = 0
        else: 
            ones_pct = float(num_ones / len(y))
            zeros_pct = float(num_zeros / len(y))
            entropy = - ones_pct * np.log2(ones_pct) - zeros_pct * np.log2(zeros_pct)
        return entropy
    
    def _majority_vote(self, y):
        num_ones = np.count_nonzero(y == '1')
        num_zeros = len(y) - num_ones
        if num_zeros > num_ones:
            vote = 0
        else:
            vote = 1  
        predictions = [vote] * len(y) 
        return predictions
        
    def _error_rate(self, predictions, y):
        err_cnt = 0
        for i in range(len(predictions)):
            if predictions[i] != int(y[i]):
                err_cnt += 1
        error_rate = err_cnt / len(predictions)
        return error_rate

if __name__ == "__main__":
    args = sys.argv
    
    # Parse every argument
    _input = args[1]
    _output = args[2]
    
    data = np.genfromtxt(_input, delimiter = '\t', dtype = None, encoding = None)
    y = data[1:][:, -1]
    ins =Inspection()
    predictions = ins._majority_vote(y)
    data_entropy = ins._entropy(y)
    data_error_rate = ins._error_rate(predictions, y)
    
    with open(_output, 'w') as f:
       f.write('entropy: ' + "%.6f" % data_entropy + '\n')
       f.write('error: ' + "%.6f" % data_error_rate)
