import numpy as np

class OptimizerFactory:
    def build(self, name):
        if name == 'sgd':
            return SGD()
        if name == 'rmsprop':
            return RMSProp()
        return SGD()


class Optimizer:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        
    def update(self, input, grad_output, weights, biases):
        pass


class SGD(Optimizer):
    def update(self, input, grad_output, weights, biases):
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        assert grad_weights.shape == weights.shape and grad_biases.shape == biases.shape
        return weights - self.learning_rate * grad_weights, biases - self.learning_rate * grad_biases


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.002, alpha=0.9):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.gw2 = None
        self.gb2 = None
        
    def update(self, input, grad_output, weights, biases):
        if self.gw2 is None:
            self.gw2 = np.zeros_like(weights)
        if self.gb2 is None:
            self.gb2 = np.zeros_like(biases)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        assert grad_weights.shape == weights.shape and grad_biases.shape == biases.shape
        eps = 1e-6
        self.gw2 = self.alpha * self.gw2 + (1.0 - self.alpha) * np.square(grad_weights)
        self.gb2 = self.alpha * self.gb2 + (1.0 - self.alpha) * np.square(grad_biases)
        return weights - np.multiply((self.learning_rate / np.sqrt(self.gw2 + eps)), grad_weights), biases - np.multiply((self.learning_rate / np.sqrt(self.gb2 + eps)), grad_biases)