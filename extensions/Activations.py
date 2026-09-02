from core.Nexus import Nexus
import numpy as np

"""
# compute refers to any operation neccesary for getting out_val

def activation(node: Nexus) -> Nexus:
    out_val = compute(node) 
    out = Nexus(out_val)
    out._childern = {node}

    def _backward():
        activation_prime = compute(node.value)
        node.grads += out.grads*activation_prime

    out._backward = _backward

    return out
"""

class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, node: Nexus) -> Nexus:
        return self.forward(node)

    def forward(self, node: Nexus) -> Nexus:
        out_val = np.maximum(0, node.value)
        out = Nexus(out_val)
        out._children = {node}

        def _backward():
            relu_prime = (node.value > 0).astype(np.float32)
            node.grads += out.grads*relu_prime

        out._backward = _backward

        return out


class Sigmoid:
    def __int__(self):
        pass

    def __call__(self, node: Nexus) -> Nexus:
        return self.forward(node)
    
    def forward(self, node: Nexus) -> Nexus:
        out_val = 1/(1 + np.exp(-np.clip(node.value, -500, 500)))
        out = Nexus(out_val)
        out._children = {node}

        def _backward():
            sigmoid_prime = out.value*(1-out.value)
            node.grads += out.grads*sigmoid_prime

        out._backward = _backward

        return out


class LReLU:
    def __init__(self):
        pass

    def __call__(self, node: Nexus) -> Nexus:
        return self.forward(node)
    
    def forward(self, node: Nexus, alpha=0.01) -> Nexus:
        out_val = np.maximum(alpha*node.value, node.value)
        out = Nexus(out_val)
        out._children = {node}

        def _backward():
            lrelu_prime = np.where(node.value > 0, 1, alpha)
            node.grads += out.grads*lrelu_prime

        out._backward = _backward

        return out


class SiLU:
    def __int__(self):
        pass

    def __call__(self, node: Nexus) -> Nexus:
        return self.forward(node)
    
    def forward(self, node: Nexus) -> Nexus:
        sig = 1/(1 + np.exp(-np.clip(node.value, -500, 500)))
        out_val = node.value*sig
        out = Nexus(out_val)
        out._children = {node}

        def _backward():
            silu_prime = out.value + sig*(1-out.value)
            node.grads += out.grads*silu_prime

        out._backward = _backward

        return out

class Softmax:
    def __int__(self):
        pass

    def __call__(self, node: Nexus) -> Nexus:
        return self.forward(node)
    
    def forward(self, node: Nexus):
        logits = node.value - np.max(node.value, anodeis=-1, keepdims=True)
        enodeps = np.exp(logits)
        out_val = enodeps/np.sum(enodeps, anodeis=-1, keepdims=True)
        out = Nexus(out_val)
        out._children = {node}

        def _backward():
            softmanode_prime = np.sum(out.grads*out.value, anodeis=-1, keepdims=True)
            node.grads += out.value*(out.grads - softmanode_prime)

        out._backward = _backward

        return out