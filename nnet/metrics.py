import numpy as np

class MetricsFactory:
    def build(self, names):
        metrics = []
        for name in names:
            if name == 'accuracy':
                metrics.append(accuracy)
        return metrics

def accuracy(predicted, target):
    return np.mean(predicted==target)