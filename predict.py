# predict.py
import numpy as np

MODEL_PATH = "model.npz"

def load_model(path=MODEL_PATH):
    data = np.load(path)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    return W1, b1, W2, b2

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# The required function name: prediction
# Input: features -> 1D numpy array of length 4096 (already scaled between 0..1)
# Output: estimated count (we return predicted class 0/1). We also return probability.
def prediction(features):
    """
    features: 1D numpy array shape (4096,) values scaled 0..1
    Returns: integer 0 or 1 (predicted class)
    """
    W1, b1, W2, b2 = load_model(MODEL_PATH)
    x = features.reshape(1, -1)  # (1, 4096)
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    prob = sigmoid(z2)[0, 0]
    label = int(prob >= 0.5)
    return label  # "estimated count" as requested

# helper: predict from image file 
def predict_from_image(path, image_size=(64,64)):
    from PIL import Image
    import numpy as _np
    img = Image.open(path).convert("L").resize(image_size)
    arr = _np.asarray(img, dtype=_np.float32) / 255.0
    feats = arr.flatten()
    return prediction(feats)


# image file
label = predict_from_image("KimJongUn.jpg")
print("Predicted:", label)
if label == 1:
    print("The face **IS** Kim Jong Un.")
else:
    print("The face is **NOT** Kim Jong Un.")