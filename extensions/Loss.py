from core.Nexus import Nexus
import numpy as np

def MSError(target: Nexus, prediction: Nexus  ) -> Nexus:
    diff = prediction.value - target.value
    
    loss = 0.5*np.mean(np.square(diff))
    out = Nexus(loss)
    out._children = {prediction}

    def _backward():
        mse_prime = diff/prediction.value.size
        prediction.grads += prediction._handle_broadcast(out.grads*mse_prime, prediction.dimension)

    out._backward = _backward
    
    return out

def MAError(target: Nexus, prediction: Nexus) -> Nexus:
    mod = prediction - target
    error = 0.5*(np.mean(np.abs(mod)))

    out = Nexus(error)
    out._children = {prediction}

    def _backward():
        mae_prime = np.sign(mod)/prediction.value.size
        prediction.grads += prediction._handle_broadcast(out.grads*mae_prime, prediction.dimension)

    out._backward = _backward

    return out

def BinaryCrossEntropyLoss(target: Nexus, logits: Nexus) -> Nexus:
    """
    Numerically stable Binary Cross-Entropy Loss combined with an internal Sigmoid layer.
    Expects raw unactivated logit of shape (batch_size, 1) and binary targets (0.0 or 1.0).
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits.value, -500, 500)))
    batch_size = logits.value.shape[0]
    loss_val = np.mean(np.maximum(logits.value, 0) - logits.value * target.value + np.log(1.0 + np.exp(-np.abs(logits.value))))

    out = Nexus(loss_val)
    out._children = {logits}

    def _backward():
        local_derivative = (probs - target.value) / batch_size
        logits.grads += logits._handle_broadcast(out.grads * local_derivative, logits.dimension)

    out._backward = _backward
    return out

def CategoricalCrossEntropyLoss(targets: Nexus, logits: Nexus) -> Nexus:
    """"
    Parameters:
    - logits: Nexus node of shape (batch_size, num_classes) -> Raw outputs from last Linear layer
    - targets: Nexus node of shape (batch_size, num_classes) -> One-hot encoded labels
    """
    shifted_logits = logits.value - np.max(logits.value, axis=-1, keepdims=True)
    exps = np.exp(shifted_logits)
    probabilities = exps / np.sum(exps, axis=-1, keepdims=True)
    
    batch_size = logits.value.shape[0]
    clipped_probs = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss_val = -np.sum(targets.value * np.log(clipped_probs)) / batch_size
    
    out = Nexus(loss_val, _children=(logits,), _op='SoftmaxCrossEntropyLoss')

    def _backward():
        local_derivative = (probabilities - targets.value) / batch_size        
        logits.grads += logits._handle_broadcast(out.grads * local_derivative, logits.dimension)

    out._backward = _backward

    return out