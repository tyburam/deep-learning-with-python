import numpy as np

class LossFactory:
    def build(self, name):
        return SoftmaxCrossentropyWithLogitsLoss()

class LossFunction:
    def __init__(self):
        pass
    
    def calculate(self, predicted, target):
        pass
    
    def gradient(self, predicted, target):
        pass
    

class SoftmaxCrossentropyWithLogitsLoss(LossFunction):
    def calculate(self, predicted, target):
        """Compute crossentropy from predicted[batch,n_classes] and ids of correct answers"""
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        logits_for_answers = predicted[np.arange(len(predicted)), target]    
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(predicted),axis=-1))    
        return xentropy
    
    def gradient(self, predicted, target):
        """Compute crossentropy gradient from predicted[batch,n_classes] and ids of correct answers"""
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        ones_for_answers = np.zeros_like(predicted)
        ones_for_answers[np.arange(len(predicted)), target] = 1    
        softmax = np.exp(predicted) / np.exp(predicted).sum(axis=-1,keepdims=True)    
        return (- ones_for_answers + softmax) / predicted.shape[0]