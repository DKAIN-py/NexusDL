from core.Nexus import Nexus
import numpy as np

"""
# compute refers to any operation neccesary for getting forward pass

class Layer:
    def __init__(self, in_features: int, out_features: int):
        self.weights = Nexus(np.random.randn(in_features, out_features))
        self.bias = Nexus(np.zeros(shape=(1,out_features)))
    
    def __call__(self, x: Nexus) -> Nexus:
        return compute(x, self.weights, self.bias)

    def _parameters(self):
        return [self.weights, self.bias]

"""

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.weights = Nexus(np.random.randn(in_features, out_features))
        self.bias = Nexus(np.zeros(shape=(1,out_features)))
    
    def __call__(self, x: Nexus) -> Nexus:
        return (x@self.weights + self.bias)

    def _parameters(self):
        return [self.weights, self.bias]

