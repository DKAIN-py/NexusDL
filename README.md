# Nexus-Autograd

A minimalist, production-pattern, define-by-run deep learning engine built completely from scratch using NumPy. 

`Nexus` implements an automatic differentiation graph framework utilizing **operator overloading** and dynamic graph reconstruction via reverse-mode automatic differentiation.

---

## Design Philosophy

1. **Primitives over Bloat:** The core engine (`Nexus`) only manages raw tensor multidimensional transformations, dynamic DAG mapping, and memory-safe topological sorting. 

2. **Bring Your Own Math (BYOM):** Activations and layers sit completely decoupled from the core structure as pluggable extensions, exposing a strict interface contract for downstream developers.

---

## System Architecture

Nexus builds its computational map dynamically as operations execute on variables, mapping memory dependencies explicitly behind the scenes.

* **Dynamic DAG Tracing:** Every `Nexus` node acts as a pointer-tracker using Python `set` collections (`_children`) to enforce pointer uniqueness. This completely eliminates duplicate evaluations and infinite looping bugs during Reverse Depth-First Search (DFS).
* **Automated Chain Rule:** The local derivative calculations are decoupled and bundled into closures attached directly to output nodes (`out._backward`). Calling `.backward()` automatically runs a topological sort and orchestrates gradient accumulation natively.

---

## Supported Primitives & Activations

### Mathematical Operators
* **Binary Primitives:** `+` (`__add__`), `-` (`__sub__`), `*` (`__mul__`), `/` (`__truediv__`)
* **Matrix Calculus:** `@` (`__matmul__`) with explicit automatic transposition mapping
* **Unary / Exponentials:** `**` (`__pow__`) with native scalar power rules

### Pluggable Activations
* **ReLU / LeakyReLU:** Clean element-by-element conditional tracking
* **Sigmoid / SiLU (Swish):** Optimized for numerical stability, utilizing native analytical derivatives to mitigate division-by-zero (`NaN`) errors.
* **Vectorized Softmax:** Batch-safe Jacobian trace formulation executed loop-free over the last axis.

---

## Usage Example

Here is how seamlessly you can construct a forward execution graph and extract exact analytical gradients down to the leaf nodes:

```python
import numpy as np
from core.Nexus import Nexus
from extensions.Activations import ReLU

# 1. Initialize variables (automatically standardized to Float32 NumPy arrays)
x = Nexus([[1.0, 2.0], [3.0, 4.0]])
w = Nexus([[0.5, -0.1], [0.2, 0.8]])
b = Nexus([0.1, 0.1])

# 2. Forward pass (dynamically generates the operational graph)
z = (x @ w) + b
output = ReLU(z)

# 3. Trigger the Topological Engine
output.backward()

# 4. Extract pre-computed gradients for your optimizer step
print(w.grads)
print(b.grads)
```
```
autograd_engine/
│
├── core/                         # The Engine Core
│   ├── __init__.py
│   ├── Nexus.py                  # Node wrapping, broadcast handling, dunder overrides
│   └── model.py                  # Upcoming: Structural Sequential wrapper
│
├── extensions/                   # Pluggable Math Stack
│   ├── __init__.py
│   ├── activations.py            # Vectorized activation derivatives
│   ├── losses.py                 # Upcoming: MSE, Cross-Entropy Loss
│   └── optimizers.py             # Upcoming: SGD, Adam gradient steppers
│
└── examples/                     # Sandboxed Verification Loops
    └── custom_extension_demo.py
```