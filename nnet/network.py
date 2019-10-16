import numpy as np

from .layers import Layer
from .losses import LossFactory
from .optimizers import OptimizerFactory
from .metrics import MetricsFactory

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.metrics = []
        self.metric_names = ()
        
    def append(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)
        
    def compile(self, optimizer='sgd', loss='crossentropy', metrics=[]):
        self.loss = LossFactory().build(loss)
        self.metric_names = list(set(metrics))
        self.metrics = MetricsFactory().build(self.metric_names)

        for i in range(len(self.layers)):
            self.layers[i].set_optimizer(OptimizerFactory().build(optimizer))
        
    def forward(self, X):
        """
        Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer. 
        Make sure last activation corresponds to network logits.
        """
        
        activations = []
        input = X
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
            activations.append(input)

        assert len(activations) == len(self.layers)
        return activations
    
    def predict(self, X):
        """
        Compute network predictions.
        """
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)
    
    def epoch(self, X, y, backprop=True):
        """
        Train your network on a given batch of X and y.
        You first need to run forward to get all layer activations.
        Then you can run layer.backward going from last to first layer.

        After you called backward for all layers, all Dense layers have already made one gradient step.
        """

        # Get the layer activations
        layer_activations = self.forward(X)
        layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss_val = self.loss.calculate(logits, y)
        loss_grad = self.loss.gradient(logits, y)

        if backprop:
            grad = loss_grad
            for i in range(len(self.layers)-1,-1,-1):
                grad = self.layers[i].backward(layer_inputs[i], grad)

        return np.mean(loss_val)
    
    def fit(self, epochs, X, y, X_val=None, y_val=None):
        history = {'loss': [], 'val_loss': []}
        for _ in range(epochs):
            history['loss'].append(self.epoch(X, y))
            if X_val is not None:
                history['val_loss'].append(self.evaluate(X_val, y_val))

            if self.metrics is not None:
                for i in range(len(self.metrics)):
                    if self.metric_names[i] not in history:
                        history[self.metric_names[i]] = []
                        history['val_' + self.metric_names[i]] = []
                    history[self.metric_names[i]].append(self.metrics[i](self.predict(X), y))
                    if X_val is not None:
                        history['val_' + self.metric_names[i]].append(self.metrics[i](self.predict(X_val), y_val))

        return history
    
    def evaluate(self, X, y):
        return self.epoch(X, y, False)