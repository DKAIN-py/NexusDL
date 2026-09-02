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
    
    def export_weights(self, filepath: str):
        import json, os
        model_meta = []
        param_arrays = {}

        for idx, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            meta_entry = {"index":idx, "type":layer_name}

            if hasattr(layer, "weights"):
                w_file = f"layer_{idx}_{layer_name}_W.bin"
                b_file = f"layer_{idx}_{layer_name}_B.bin"

                meta_entry["weights_file"] = w_file
                meta_entry["bias_file"] = b_file
                meta_entry["shape_W"] = layer.weights.value.shape
                meta_entry["shape_B"] = layer.weights.value.shape

                param_arrays[w_file] = layer.weights.value
                param_arrays[b_file] = layer.bias.value

            model_meta.append(meta_entry)
        
        manifest_path = os.path.join(filepath, "model_manifest.json")
        os.makedirs(filepath, exist_ok=True)

        with open(manifest_path, "w") as f:
            json.dump(model_meta, f, indent=4)
        
        for filename, array in param_arrays.items():
            full_path = os.path.join(filepath, filename)
            array.tofile(full_path)