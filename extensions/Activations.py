from core.Nexus import Nexus
import numpy as np

def ReLU(node: Nexus) -> Nexus:
    out_val = np.maximum(0, node.value)
    out = Nexus(out_val)
    out._children = {node}

    def _backward():
        relu_prime = (node.value > 0).astype(np.float32)
        node.grads += out.grads*relu_prime

    out._backward = _backward

    return out

def sigmoid(node: Nexus) -> Nexus:
    out_val = 1/(1 + np.exp(-np.clip(node.value, -500, 500)))
    out = Nexus(out_val)
    out._children = {node}

    def _backward():
        sigmoid_prime = out.value*(1-out.value)
        node.grads += out.grads*sigmoid_prime

    out._backward = _backward

    return out

def LReLU(node: Nexus, alpha=0.01) -> Nexus:
    out_val = np.maximum(alpha*node.value, node.value)
    out = Nexus(out_val)
    out._children = {node}

    def _backward():
        lrelu_prime = np.where(node.value > 0, 1, alpha)
        node.grads += out.grads*lrelu_prime
    
    out._backward = _backward

    return out

def SiLU(node: Nexus) -> Nexus:
    sig = 1/(1 + np.exp(-np.clip(node.value, -500, 500)))
    out_val = node.value*sig
    out = Nexus(out_val)
    out._children = {node}

    def _backward():
        silu_prime = out.value + sig*(1-out.value)
        node.grads += out.grads*silu_prime

    out._backward = _backward

    return out

def Softmax(node: Nexus):
    logits = node.value - np.max(node.value, axis=-1, keepdims=True)
    exps = np.exp(logits)
    out_val = exps/np.sum(exps, axis=-1, keepdims=True)
    out = Nexus(out_val)
    out._children = {node}

    def _backward():
        softmax_prime = np.sum(out.grads*out.value, axis=-1, keepdims=True)
        node.grads += out.value*(out.grads - softmax_prime)
    
    out._backward = _backward

    return out