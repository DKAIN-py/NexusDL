from .Nexus import Nexus

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self) -> list[Nexus]:
        all_para = []

        for layer in self.layers:
            if hasattr(layer, "_parameters"):
                all_para.extend(layer._parameters())
        
        return all_para