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

class Adam:
    def __init__(self, params, lr: float=0.001, beta1: float =0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self, max_norm: float = 1.0):
        self.t+=1

        total_norm = np.sqrt(sum(np.sum(p.grads ** 2) for p in self.params if p.grads is not None))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.params:
                if p.grads is not None:
                    p.grads *= clip_coef

        for i,p in enumerate(self.params):
            if p.grads is None:
                continue

            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*p.grads
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(p.grads**2)

            m_hat = self.m[i]/(1- self.beta1**self.t)
            v_hat = self.v[i]/(1- self.beta2**self.t)

            p.value -= self.lr*m_hat/(np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grads = np.zeros_like(p.value, dtype=np.float32)