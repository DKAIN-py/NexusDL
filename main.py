from core.Nexus import Nexus, no_grad
from core.model import Sequential
from extensions.Layers import Linear
from extensions.Activations import ReLU
from extensions.Loss import BinaryCrossEntropyLoss
from extensions.Optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
 
import pandas as pd
# 1. Load the dataframe
raw_df = pd.read_csv('Iris.csv')

# 2. Map species text to clean 0.0 and 1.0 floats directly
raw_df['Species'] = raw_df['Species'].replace({'Iris-setosa': 0.0, 'Iris-versicolor': 1.0})

# 3. CRITICAL: Select ONLY the 4 valid feature columns by name.
# This leaves the 'Id' and any weird index columns behind!
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# 4. Extract as explicit float32 arrays right from the source
x = raw_df[feature_columns].values.astype(np.float32)
y = raw_df['Species'].values.astype(np.float32).reshape(-1, 1)

# 5. Split your data splits safely
x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
    x, y, train_size=0.8, random_state=50
)

# 6. Wrap training data directly into your clean Nexus engine nodes
x_train = Nexus(x_train_raw)
y_train = Nexus(y_train_raw)
print(f"Sanitized x_train shape: {x_train.value.shape} | dtype: {x_train.value.dtype}")
print(f"Sanitized y_train shape: {y_train.value.shape} | dtype: {y_train.value.dtype}")

model = Sequential(
    Linear(in_features=4, out_features=8),
    ReLU,
    Linear(8,1),
)

optim = SGD(model.parameters(), lr=0.01)


epochs = 100
losses = []

for i in range(epochs):

    optim.zero_grad()

    y_pred = model.forward(x_train)
    loss = BinaryCrossEntropyLoss(y_train, y_pred)
    losses.append(loss.value.item())
    if i%10==0:
        print(f"Epoch: {i} and loss: {loss.value.item()}")
    loss.backward()
    optim.step()


correct = 0
with no_grad():
    for i, data in enumerate(x_test_raw):
        data_node = Nexus(data.reshape(1,-1))
        logits = model.forward(data_node)
        probs = 1.0 / (1.0 + np.exp(-logits.value))
        predicted_class = np.round(probs)

        if int(predicted_class.item()) == int(y_test_raw[i][0]):
            correct += 1

print(f"\nNumber of correct predictions: {correct} / {len(x_test_raw)}")
print(f"Accuracy: {(correct / len(x_test_raw)) * 100:.2f}%")

import matplotlib.pyplot as plt 

plt.plot(range(epochs), losses)
plt.title("BCE Optimization Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()