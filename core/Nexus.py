import numpy as np

class Nexus:
    _track_graph = True
    def __init__(self, value):
        self.value = np.array(value, dtype=np.float32) if not isinstance(value, np.ndarray) else value.astype(np.float32)
        self.dimension = self.value.shape
        self.grads = np.zeros(shape=self.dimension, dtype=np.float32)
        self._children = set()
        
        self._backward = lambda: None
    
    def __add__(self, other):
        if not isinstance(other, Nexus):
            other = Nexus(other)
        out_val = self.value + other.value
        children = (self, other) if Nexus._track_graph else ()
        out = Nexus(out_val)
        out._children = children

        if Nexus._track_graph:
            def _backward():
                self.grads += self._handle_broadcast(out.grads, self.dimension)
                other.grads += self._handle_broadcast(out.grads, other.dimension)

            out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Nexus) else Nexus(other)

        out = Nexus(self.value - other.value)
        out._children = {self, other}

        if Nexus._track_graph:
            def _backward():
                self.grads += self._handle_broadcast(out.grads, self.dimension)
                other.grads += self._handle_broadcast(-out.grads, other.dimension)      
            out._backward = _backward

        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Nexus) else Nexus(other)
        return other.__sub__(self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Nexus) else Nexus(other)

        out = Nexus(np.multiply(self.value,other.value))
        out._children = {self, other}

        if Nexus._track_graph:

            def _backward():
                self.grads += self._handle_broadcast(out.grads*other.value, self.dimension)
                other.grads += self._handle_broadcast(out.grads*self.value, other.dimension)

            out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Nexus) else Nexus(other)

        out = Nexus(np.divide(self.value, other.value))
        out._children = {self, other}

        if Nexus._track_graph:

            def _backward():
                raw_grads_self =  np.divide(out.grads, other.value)
                raw_grads_other = -np.divide( np.multiply(out.grads, self.value), np.square(other.value))
                self.grads += self._handle_broadcast(raw_grads_self, self.dimension)
                other.grads += self._handle_broadcast(raw_grads_other, other.dimension)

            out._backward = _backward

        return out
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Nexus(np.power(self.value, power))
        out._children = {self}

        if Nexus._track_graph:

            def _backward():
                self.grads += out.grads*(power*(self.value**(power-1)))

            out._backward = _backward

            return out
    
    def __rpow__(self, power):
        return self.__pow__(power)
    
    def __matmul__(self, other):
        if not isinstance(other, Nexus):
            other = Nexus(other)

        out_val = np.matmul(self.value, other.value)
        children = (self, other) if Nexus._track_graph else ()
        out = Nexus(out_val)
        out._children = children

        if Nexus._track_graph:
        
            def _backward():
                raw_grad_self = np.matmul(out.grads, other.value.T)
                raw_grad_other = np.matmul(self.value.T, out.grads)

                self.grads += self._handle_broadcast(raw_grad_self, self.dimension)
                other.grads += self._handle_broadcast(raw_grad_other, other.dimension)

            out._backward = _backward

        return out
    
    def __rmatmul__(self, other):
        other = other if isinstance(other, Nexus) else Nexus
        return other.__matmul__(self)

    def __repr__(self):
        return f"Nexus(value={self.value}, grads={self.grads})"

    def sin(self):
        out = Nexus(np.sin(self.value))
        out._children = {self}

        if Nexus._track_graph:
            
            def _backward():
                self.grads += self._handle_broadcast(out.grads*np.cos(self.value), self.dimension)

            out._backward = _backward

        return out
    
    def cos(self):
        out = Nexus(np.cos(self.value))
        out._children = {self}

        if Nexus._track_graph:

            def _backward():
                cos_prime = -np.sin(self.value)
                self.grads += self._handle_broadcast(out.grads*cos_prime, self.dimension)

            out._backward = _backward

        return out
    
    def log2(self):
        out = Nexus(np.log2(self.value))
        out._children = {self}

        if Nexus._track_graph:

            def _backward():
                self.grads += self._handle_broadcast(out.grads*((1/self.value)*np.log(2)), self.dimension)

            out._backward = _backward

        return out
    
    def log(self):
        out = Nexus(np.log(self.value))
        out._children = {self}

        if Nexus._track_graph:

            def _backward():
                self.grads += self._handle_broadcast(out.grads*(1/self.value), self.dimension)

            out._backward = _backward

        return out

    def exp(self):
        out = Nexus(np.exp(self.value))
        out._children = {self}

        if Nexus._track_graph:

            def _backward():
                self.grads += self._handle_broadcast(out.grads*(out.value), self.dimension)

            out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: Nexus):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        self.grads = np.ones_like(self.value, dtype=np.float32)

        for node in reversed(topo):
            node._backward()

    def apply(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _handle_broadcast(self, grad, target_shape):
        # Used to handle numpy broadcasting
        if grad.shape == target_shape:
            return grad
        
        # find the axis expanded by numpy
        lead = grad.ndim - len(target_shape)
        if lead > 0:
            lead_axis = tuple(range(lead))
            # sum along that axis to remove the axis
            grad = grad.sum(axis=lead_axis)

        # finds if there was any dimension = 1
        keep_axis = tuple(i for i, dim in enumerate(target_shape) if dim==1)
        if keep_axis:
            grad = grad.sum(axis=keep_axis, keepdims=True)

        return grad


    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        allowed_data_types = (list, np.ndarray, int, float)
        if not isinstance(value, allowed_data_types):
            raise ValueError("Input should be either a python list or a numpy ndarray!")

        if isinstance(value, (list, int, float)):
            value = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            value = value.astype(np.float32)

        self._value = value


class no_grad:
    def __enter__(self):
        self.prev = Nexus._track_graph
        Nexus._track_graph = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Nexus._track_graph = self.prev