import numpy as np

def accuracy(predicted, target):
    return np.mean(predicted==target)