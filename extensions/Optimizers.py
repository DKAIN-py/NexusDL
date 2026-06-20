from core.Nexus import Nexus
import numpy as np

"""
# compute refers to any operation neccesary for updating p.value

class Optimizer:
    def __init__(self, parameters: list[Nexus], lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for p in self.parameters:
            p.grads = np.zeros_like(p.value, dtype=np.float32)
    
    def step(self):
        for p in self.parameters:
            grads_value = p.grads.value if isinstance(p.grads, Nexus) else p.grads
            p.value = compute(p.value, self.lr, grads_value, *args, **kwargs)        
"""


class SGD:
    def __init__(self, parameters: list[Nexus], lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grads = np.zeros_like(p.value, dtype=np.float32)
    
    def step(self):
        for p in self.parameters:
            grads_value = p.grads.value if isinstance(p.grads, Nexus) else p.grads
            p.value = p.value - self.lr*grads_value