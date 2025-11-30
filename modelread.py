import numpy as np

data = np.load("model.npz")

print(data.files)

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

print(W1.shape, b1.shape, W2.shape, b2.shape)

import numpy as np

data = np.load("model.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
