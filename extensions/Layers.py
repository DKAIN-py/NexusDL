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

from core.helpers import im2col
class Conv2D:
    def __init__(self, kernel_size: int | tuple, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        bound = 1.0/np.sqrt(in_channels*kh*kw)

        self.weights = Nexus(np.random.uniform(-bound, bound, (out_channels, in_channels*kh*kw)))
        self.bias = Nexus(np.zeros((out_channels, 1)))


    def __call__(self, x: Nexus) -> Nexus:
        N, C, H, W = x.dimension
        kh, kw = self.kernel_size

        H_out = (H+2*self.padding-kh)//self.stride+1
        W_out = (W+2*self.padding-kw)//self.stride+1

        x_col = im2col(x,kh,kw,self.stride,self.padding)


        out_mat = self.weights@x_col

        out_4d = out_mat.reshape((self.out_channels, N,H_out,W_out)).transpose(1,0,2,3)

        bias_4d = self.bias.reshape((1, self.out_channels, 1, 1))

        return out_4d + bias_4d

    def _parameters(self):
        return [self.weights, self.bias]


from core.helpers import maxpool2d
class MaxPool2D:
    def __init__(self, kernel_size: int | tuple, stride: int = None):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: Nexus) -> Nexus:
        return maxpool2d(x, self.kernel_size, self.stride)

    def _parameters(self):
        return []


class Flatten:
    def __call__(self, x: Nexus) -> Nexus:
        N = x.dimension[0]
        return x.reshape((N, -1))

    def parameters(self):
        return []