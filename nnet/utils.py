import numpy as np

def from_categorical(data):
    return np.argmax(data, axis=1)